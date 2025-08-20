import os
import json
import re
import requests
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- Config from environment (Render injects these) ----
HF_TOKEN = os.getenv("HF_API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct:novita")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

if not HF_TOKEN:
    # Fail fast so you notice misconfiguration
    raise RuntimeError("HF_API_KEY is not set (Render → Service → Settings → Environment).")

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
HF_URL = "https://router.huggingface.co/v1/chat/completions"

# ---- FastAPI app + CORS ----
app = FastAPI(title="StyleCheck Word Generator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request schema from your website ----
class GenerateRequest(BaseModel):
    apparel_type: Optional[str] = None
    attributes: Optional[str] = None  # e.g., "Brand: Zara, Fit: Slim, Material: Cotton, Color: Blue"

# ---- Helpers ----
def sanitize_and_parse_json(s: str) -> Dict[str, Any]:
    # Strip code fences if model adds ```json ... ```
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))

def build_prompt(attributes_text: str) -> str:
    return f"""
You are a fashion product naming and description expert.

Based on the following clothing attributes, create:

1. A 3-word product name
2. A 5-word product name
3. An 8-word product name
4. A short description (2 sentences)
5. A long description (detailed, include 3-4 bullet points)

Clothing Attributes:
{attributes_text}

Return ONLY valid JSON (no markdown fences), exactly:
{{
  "three_word_name": "...",
  "five_word_name": "...",
  "eight_word_name": "...",
  "short_description": "...",
  "long_description": ["point 1", "point 2", "point 3"]
}}
""".strip()

def call_hf(prompt: str) -> Dict[str, Any]:
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 380,
        "stream": False
    }
    r = requests.post(HF_URL, headers=HEADERS, json=payload, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    content = r.json()["choices"][0]["message"]["content"]
    try:
        return sanitize_and_parse_json(content)
    except Exception as e:
        # Bubble up raw text so you can debug formatting, while keeping API predictable
        raise HTTPException(status_code=502, detail=f"Invalid JSON from model: {e}\nRaw: {content}")

# ---- Routes ----
@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_ID}

@app.post("/generate")
def generate(req: GenerateRequest):
    if not req.attributes:
        raise HTTPException(status_code=400, detail="Provide 'attributes' string from your website.")
    prompt = build_prompt(req.attributes)
    result = call_hf(prompt)
    return {"generated": result}
