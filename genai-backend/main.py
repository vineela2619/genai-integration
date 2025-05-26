from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    new_message: str

class ChatResponse(BaseModel):
    message: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="API key not configured")

        # Configure the Gemini API
        genai.configure(api_key=api_key)

        # Initialize the model
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Prepare conversation history
        conversation_history = []
        for msg in request.messages:
            conversation_history.append({
                "role": msg.role,
                "parts": [{"text": msg.content}]
            })

        # Add new user message
        conversation_history.append({
            "role": "user",
            "parts": [{"text": request.new_message}]
        })

        # Get response from Gemini
        response = model.generate_content(
            contents=conversation_history,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=150
            )
        )

        # Extract the response text
        if not response.parts:
            raise HTTPException(status_code=500, detail="No response parts returned by Gemini API")

        return ChatResponse(message=response.parts[0].text.strip())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)