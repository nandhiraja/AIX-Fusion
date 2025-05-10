from fastapi import FastAPI, HTTPException,File, UploadFile
from fastapi.responses import JSONResponse
import easyocr
import numpy as np
from fastapi import  Query
from phi.agent import Agent, RunResponse
from phi.model.groq import Groq
from phi.tools.baidusearch import BaiduSearch
from phi.tools.serpapi_tools import SerpApiTools
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from fastapi.responses import StreamingResponse
from huggingface_hub import InferenceClient
import os
import requests
import io
from PIL import Image
from fastapi.responses import Response
import uvicorn
import enum

from dotenv import load_dotenv
load_dotenv()


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

app = FastAPI()




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#--------------------------------------------------------------------------------------------------------------------------------------------------

                    # Routing of the Image Generation API

                    
HF_API_KEY = os.getenv("HF_API_KEY")


API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-dev"

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

class ImageRequest(BaseModel):
    prompt: str

def query(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error generating image")
    return response.content

@app.post("/generate-image")
def generate_image(request: ImageRequest):
    image_bytes = query({"inputs": request.prompt})
    return Response(content=image_bytes, media_type="image/png")


#--------------------------------------------------------------------------------------------------------------------------------------------------

                 # Routing of the Video Generation API


if not HF_API_KEY:
    raise ValueError("API key is missing. Set HF_API_KEY as an environment variable.")

# Initialize Hugging Face client
client = InferenceClient(provider="replicate", api_key=HF_API_KEY)

class VideoRequest(BaseModel):
    prompt: str

@app.post("/generate_video/")
async def generate_video(request: VideoRequest):
    try:
        # Generate video from text
        video_data = client.text_to_video(
            request.prompt,
            model="Wan-AI/Wan2.1-T2V-14B"
        )

        # Convert video data into a streamable response
        video_stream = io.BytesIO(video_data)

        return StreamingResponse(video_stream, media_type="video/mp4")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

#--------------------------------------------------------------------------------------------------------------------------------------------------
                               # Routing of the OCR API



reader = easyocr.Reader(['ch_sim', 'en'])
@app.post("/ocr/")
async def ocr_text(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Convert image to NumPy array
        image_np = np.array(image)

        # Perform OCR
        result = reader.readtext(image_np, detail=0)

        return {"extracted_text": result}
    
    except Exception as e:
        return {"error": str(e)}


#----------------------------------------------------------------------------------------------------------------------------------------------------------------
                                             # Routing of the web Search API


groq_api_key = os.getenv("GROQ_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")


# Define individual search agents
baidu_agent = Agent(
    name="Baidu Search Agent",
    role="Search the web using Baidu",
    model=Groq(id="llama-3.1-8b-instant", api_key=groq_api_key),
    tools=[BaiduSearch()],
    instructions=["Provide key points from Baidu search results that relates user requrements. include all necessary information that need to answer user question. make sure your summary should be more alobrate and discriptive without extra data than scraped one"],
    show_tools_calls=False,
    markdown=True,
)

serp_agent = Agent(
    name="SerpAPI Search Agent",
    role="Search the web using SerpAPI",
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
    tools=[SerpApiTools(api_key=SERP_API_KEY)],
    instructions=["Provide key points from SerpAPI search results that relates user requrements. include all necessary information that need to answer user question.make sure your summary should be more alobrate and discriptive without extra data than scraped one"],
    show_tools_calls=False,
    markdown=True,
)

# Multi-agent system combining Baidu and SerpAPI results
multi_ai_agent = Agent(
    team=[baidu_agent,serp_agent],  # Using both agents
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
    instructions=[
        "Analyze and summarize the information from multiple sources.",
        "you must Give the content what the user need donot give any other response",
        "Format the response neatly, prioritizing the most reliable sources.","any other activities are not allowed",
    ],
     show_tools_calls=False,
    markdown=True,
)


@app.get("/search")
def search(query: str = Query(..., title="Search Query", description="Enter the search term")):
    """
    Perform a multi-source web search using Baidu and SerpAPI.
    """
    # query = f""" 
 
    #         {query}
    #         """

    # baidu_agent.print_response("when tariff comes in effect in india")
    # serp_agent.print_response("who won yesterday ipl match between rcb vs gt")
    # multi_ai_agent.print_response("who won yesterday ipl match between rcb vs gt")

    response= multi_ai_agent.run(query)
    
    return {"query": query, "response": response.content}


#---------------------------------------------------------------------------------------------------------------------------------------------------

                                                    # Routing of the Text Summarization API


# Define enums for summary options
class SummaryLength(str, enum.Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

class SummaryStyle(str, enum.Enum):
    CONCISE = "concise"
    DETAILED = "detailed"
    BULLETS = "bullets"

# Define request model
class SummarizationRequest(BaseModel):
    content: str
    length: SummaryLength = SummaryLength.SHORT
    style: SummaryStyle = SummaryStyle.CONCISE

# Initialize Groq LLM
llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
output_parser = StrOutputParser()

# Define summarization prompt template
def get_prompt_template(length: SummaryLength, style: SummaryStyle):
    # Configure length instructions
    length_instruction = {
        SummaryLength.SHORT: "Create a very concise summary in 2-3 sentences.",
        SummaryLength.MEDIUM: "Create a moderately detailed summary in 4-6 sentences.",
        SummaryLength.LONG: "Create a comprehensive summary that covers all main points."
    }
    
    # Configure style instructions
    style_instruction = {
        SummaryStyle.CONCISE: "Use clear, direct language focusing only on the most essential information.",
        SummaryStyle.DETAILED: "Include specific details, examples, and nuances from the original text.",
        SummaryStyle.BULLETS: "Format the summary as bullet points highlighting key information."
    }
    
    return ChatPromptTemplate.from_template(
        f"""
        You are an advanced AI model specializing in text summarization.
        Your goal is to create accurate summaries while preserving the key information and context of the original text.
        
        SUMMARY LENGTH: {length_instruction[length]}
        SUMMARY STYLE: {style_instruction[style]}
        
        Follow these rules while summarizing:
        1) Ensure the summary is coherent and easy to understand.
        2) Avoid adding any new information or making assumptions.
        3) Remove redundant or repetitive details.
        4) Keep the tone neutral unless instructed otherwise.
        5) If the input text is highly technical, maintain its terminology but make it accessible.
        
        Here is your content to summarize:
        
        {{input}}
        """
    )

# Summarization function using the recommended RunnableSequence approach
def response_summarization(content: str, length: SummaryLength, style: SummaryStyle) -> str:
    prompt = get_prompt_template(length, style)
    
    # Create a runnable sequence: prompt | llm | output_parser
    chain = prompt | llm | output_parser
    
    try:
        # Invoke the chain with the content
        return chain.invoke({"input": content})
    except Exception as e:
        # Log the specific exception
        print(f"Error in summarization: {str(e)}")
        raise e

# API route for summarization
@app.post("/summarize_text")
async def summarize_text(request: SummarizationRequest):
    try:
        if not request.content:
            raise HTTPException(status_code=400, detail="No content provided")
        
        # Print log to debug
        print(f"Received request with length={request.length}, style={request.style}")
        
        summary = response_summarization(
            content=request.content,
            length=request.length,
            style=request.style
        )
        
        return {"summary": summary}
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

                                                       # Routing of the coding Api


llm1 = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

output_parser1 = StrOutputParser()
prompt1 = ChatPromptTemplate.from_template(
    """
    You are an AI coding assistant. Your task is to generate only the required code without any explanations, comments, or additional text.  
    Strictly return the code output in the specified programming language.  
    Do not include any greetings, descriptions, or explanations.  
    If multiple possible implementations exist, return only one optimal solution.  
    If no valid code can be generated, return an empty response.  
    Questions:{input}
    """
)
chain = prompt1 | llm1 | output_parser1

# Define request model
class CodeRequest(BaseModel):
    input: str

@app.post("/generate_code")
async def generate_code(request: CodeRequest):
    """
    Generate code based on the provided input prompt.
    """
    try:
        # Validate input
        if not request.input or request.input.strip() == "":
            raise HTTPException(status_code=400, detail="Input prompt cannot be empty")
        
        # Process the request
        response = chain.invoke({"input": request.input})
        
        # Return the generated code
        return {"code": response}
    except Exception as e:
        # Log the error (in a production environment, use a proper logger)
        print(f"Error generating code: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate code: {str(e)}")                                                   


#--------------------------------------------------------------------------------------------------------------------------------------------------
                                
                                # Routing of the Speech to Text API

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
                             



WHISPER_API_URL = 	"https://api.groq.com/openai/v1/audio/transcriptions"

@app.post("/speech-to-text/")
async def speech_to_text(file: UploadFile = File(...)):
    if not  groq_api_key:
        raise HTTPException(status_code=500, detail="API Key not configured on server")
        
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        logger.info(f"Processing file: {file.filename}, size: {len(audio_bytes)} bytes, content type: {file.content_type}")
        
        # Send request to Whisper API
        try:
            response = requests.post(
                WHISPER_API_URL,
                headers={"Authorization": f"Bearer { groq_api_key}"},
                files={"file": (file.filename, audio_bytes, file.content_type)},
                data={"model": "whisper-large-v3-turbo"}  # Specify the model
            )
            
            logger.info(f"Groq API Response Status: {response.status_code}")
            
            # Handle API response
            if response.status_code != 200:
                logger.error(f"Groq API Error: {response.text}")
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"Failed to process audio: {response.text}"
                )
            
            return JSONResponse(content=response.json())
            
        except requests.RequestException as req_err:
            logger.error(f"Request error: {str(req_err)}")
            raise HTTPException(status_code=500, detail=f"API request failed: {str(req_err)}")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
















































# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)






   