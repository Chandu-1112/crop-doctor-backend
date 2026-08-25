import os
import json
import uuid
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured")


# =========================================================
# GEMINI & IN-MEMORY STORAGE
# =========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)

# This list exists only in RAM while the server is running.
# Restarting or stopping the server clears it automatically.
RECENT_DIAGNOSES = []


# =========================================================
# SCHEMAS
# =========================================================

class ChatRequest(BaseModel):
    message: str


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="LeafLens AI",
    description="AI-powered crop disease detection API",
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# ROOT & HISTORY ENDPOINTS
# =========================================================

@app.get("/")
def root():
    return {
        "message": "LeafLens AI API is running"
    }


@app.get("/history")
def get_history():
    """Returns all scans made during the current server session."""
    return RECENT_DIAGNOSES


@app.delete("/history")
def clear_history():
    """Manual endpoint if you want to clear history without restarting."""
    RECENT_DIAGNOSES.clear()
    return {"message": "History cleared successfully"}


# =========================================================
# AI ASSISTANT / CHAT ENDPOINT
# =========================================================

@app.post("/chat")
async def chat_assistant(req: ChatRequest):
    """Processes user queries about crops using Gemini Flash 2.5."""
    user_query = req.message.strip()

    if not user_query:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    system_instruction = """
    You are 'Crop Doctor AI', an expert agricultural companion and plant pathologist.
    Provide concise, helpful, and friendly advice about crop diseases, symptoms, treatments, organic solutions, and prevention techniques.
    Format your responses with clear spacing and bullet points where helpful.
    Keep answers under 3 paragraphs unless detailed instructions are requested.
    """

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[system_instruction, f"User question: {user_query}"]
        )

        if not response.text:
            raise HTTPException(
                status_code=500,
                detail="AI returned an empty response."
            )

        return {"reply": response.text.strip()}

    except Exception as e:
        print("Gemini Chat Error:", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate AI response."
        )


# =========================================================
# PREDICT
# =========================================================

@app.post("/predict")
async def predict_leaf(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # CHECK FILE
    # -----------------------------------------------------

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="File type could not be determined."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )


    # -----------------------------------------------------
    # READ IMAGE
    # -----------------------------------------------------

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )


    # -----------------------------------------------------
    # PROMPT
    # -----------------------------------------------------

    prompt = """
You are an expert agricultural plant disease detection AI.

Analyze the uploaded leaf image carefully.

Identify:

1. Crop name
2. Disease name
3. Confidence percentage
4. Visible symptoms
5. Treatment recommendations
6. Prevention recommendations

If the image is not a plant leaf, clearly say so.

If the disease cannot be determined reliably,
use a low confidence value.

Return ONLY valid JSON.

Use exactly this structure:

{
    "crop": "Tomato",
    "disease": "Early Blight",
    "confidence": 85,
    "symptoms": [
        "Dark circular spots on leaves",
        "Yellowing around infected areas"
    ],
    "treatment": [
        "Remove infected leaves",
        "Apply appropriate fungicide"
    ],
    "prevention": [
        "Avoid overhead watering",
        "Maintain proper spacing"
    ]
}

Do not return markdown.
Do not return ```json.
Do not add any explanation outside the JSON.
"""


    # -----------------------------------------------------
    # SEND IMAGE TO GEMINI
    # -----------------------------------------------------

    try:

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",

            contents=[
                prompt,

                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=file.content_type
                )
            ]
        )

    except Exception as e:

        print("Gemini error:", e)

        raise HTTPException(
            status_code=500,
            detail="AI analysis failed."
        )


    # -----------------------------------------------------
    # GET RESPONSE
    # -----------------------------------------------------

    if not response.text:

        raise HTTPException(
            status_code=500,
            detail="AI returned an empty response."
        )

    result_text = response.text.strip()


    # -----------------------------------------------------
    # CLEAN JSON
    # -----------------------------------------------------

    if result_text.startswith("```json"):
        result_text = result_text[7:]

    elif result_text.startswith("```"):
        result_text = result_text[3:]

    if result_text.endswith("```"):
        result_text = result_text[:-3]

    result_text = result_text.strip()


    # -----------------------------------------------------
    # PARSE JSON
    # -----------------------------------------------------

    try:

        result = json.loads(
            result_text
        )

    except json.JSONDecodeError:

        print("Invalid AI response:")
        print(result_text)

        raise HTTPException(
            status_code=500,
            detail="AI returned invalid diagnosis data."
        )


    # -----------------------------------------------------
    # VALIDATE & ENRICH
    # -----------------------------------------------------

    required_fields = [
        "crop",
        "disease",
        "confidence",
        "symptoms",
        "treatment",
        "prevention"
    ]

    for field in required_fields:

        if field not in result:

            raise HTTPException(
                status_code=500,
                detail=f"Missing field: {field}"
            )

    # Attach session metadata
    result["id"] = str(uuid.uuid4())
    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Store in memory at the beginning of the list (most recent first)
    RECENT_DIAGNOSES.insert(0, result)


    # -----------------------------------------------------
    # RETURN
    # -----------------------------------------------------

    return result