import os
import time
from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

class LeafAnalysisResult(BaseModel):
    crop: str = Field(description="Name of the crop identified")
    disease: str = Field(description="Identified disease name or 'Healthy'")
    confidence: int = Field(description="Confidence percentage from 0 to 100")
    symptoms: List[str] = Field(description="List of visible leaf symptoms")
    treatment: List[str] = Field(description="Recommended treatment steps")
    prevention: List[str] = Field(description="Recommended prevention methods")

def analyze_image(image_data: bytes, mime_type: str) -> dict:
    prompt = "Analyze this plant leaf image. Identify the crop, possible disease, confidence score, symptoms, treatment recommendations, and prevention methods."
    
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(
                        data=image_data,
                        mime_type=mime_type,
                    ),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=LeafAnalysisResult,
                    # Disable AFC to mute warning and prevent internal tool execution conflicts
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                ),
            )

            # Safely validate and return dictionary
            if hasattr(response, "parsed") and response.parsed:
                return response.parsed.model_dump()
            
            return LeafAnalysisResult.model_validate_json(response.text).model_dump()

        except Exception as e:
            print(f"Gemini attempt {attempt + 1} failed: {e}")
            if attempt == max_attempts - 1:
                raise RuntimeError(f"Analysis failed after max retries: {str(e)}")
            time.sleep(2)