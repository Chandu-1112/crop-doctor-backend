import os
import uuid
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

# =========================================================
# CONFIGURATION & INITIALIZATION
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured")

client = genai.Client(api_key=GEMINI_API_KEY)

# Use standard production model naming
MODEL_NAME = "gemini-2.5-flash"
MAX_HISTORY_LENGTH = 100
RECENT_DIAGNOSES: List[dict] = []

# =========================================================
# SCHEMAS
# =========================================================

class ChatRequest(BaseModel):
    message: str

class DiagnosisResponse(BaseModel):
    crop: str = Field(description="Name of the crop identified")
    disease: str = Field(description="Identified disease name or 'Healthy'")
    confidence: int = Field(description="Confidence percentage (0-100)")
    symptoms: List[str] = Field(description="List of visible plant symptoms")
    treatment: List[str] = Field(description="Recommended treatment steps")
    prevention: List[str] = Field(description="Recommended prevention measures")

class FullDiagnosisResponse(DiagnosisResponse):
    id: str
    timestamp: str

# =========================================================
# FASTAPI APP SETUP
# =========================================================

app = FastAPI(
    title="LeafLens AI",
    description="AI-powered crop disease detection API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# HISTORY ENDPOINTS
# =========================================================

@app.get("/")
def root():
    return {"message": "LeafLens AI API is running"}

@app.get("/history", response_model=List[FullDiagnosisResponse])
def get_history():
    return RECENT_DIAGNOSES

@app.delete("/history")
def clear_history():
    RECENT_DIAGNOSES.clear()
    return {"message": "History cleared successfully"}

# =========================================================
# CHAT ENDPOINT
# =========================================================

@app.post("/chat")
async def chat_assistant(req: ChatRequest):
    user_query = req.message.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    system_instruction = (
        "You are 'Crop Doctor AI', an expert agricultural pathologist. "
        "Provide concise, helpful advice about crop diseases, symptoms, treatments, organic solutions, "
        "and prevention techniques. Keep responses under 3 paragraphs with bullet points where helpful."
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"User question: {user_query}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        if not response.text:
            raise HTTPException(status_code=500, detail="AI returned an empty response.")
            
        return {"reply": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Chat error: {str(e)}")

# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict", response_model=FullDiagnosisResponse)
async def predict_leaf(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    prompt = (
        "Analyze the uploaded image. Identify the crop, disease, confidence score, visible symptoms, "
        "treatment, and prevention methods. If the image is not a plant leaf, state that clearly in the fields."
    )

    try:
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=file.content_type
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DiagnosisResponse
            )
        )

        # Parse SDK output into Pydantic model natively
        parsed_data = DiagnosisResponse.model_validate_json(response.text)
        
        # Construct response object with metadata
        diagnosis_record = FullDiagnosisResponse(
            **parsed_data.model_dump(),
            id=str(uuid.uuid4()),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        # In-memory storage management with limit cap (json-safe dumping)
        RECENT_DIAGNOSES.insert(0, diagnosis_record.model_dump(mode="json"))
        if len(RECENT_DIAGNOSES) > MAX_HISTORY_LENGTH:
            RECENT_DIAGNOSES.pop()

        return diagnosis_record

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")