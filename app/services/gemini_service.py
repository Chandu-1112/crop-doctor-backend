import os
import json
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def analyze_image(image_data: bytes, mime_type: str):

    prompt = """
Analyze this plant leaf image.

Return ONLY valid JSON.
Do NOT use markdown.
Do NOT use ```json.
Do NOT add any explanation.

Use exactly this structure:

{
    "crop": "string",
    "disease": "string",
    "confidence": 0,
    "symptoms": [],
    "treatment": [],
    "prevention": []
}

Identify the crop, possible disease, confidence from 0 to 100,
visible symptoms, treatment recommendations, and prevention methods.
"""

    for attempt in range(1):

        try:

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[
                    types.Part.from_bytes(
                        data=image_data,
                        mime_type=mime_type,
                    ),
                    prompt,
                ],
            )

            text = response.text.strip()

            if text.startswith("```json"):
                text = text[7:]

            if text.startswith("```"):
                text = text[3:]

            if text.endswith("```"):
                text = text[:-3]

            text = text.strip()

            return json.loads(text)

        except Exception as e:

            print(f"Gemini attempt {attempt + 1} failed: {e}")

            if attempt == 2:
                raise

            time.sleep(2)