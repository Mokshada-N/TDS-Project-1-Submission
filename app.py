# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "sentence_transformers",
#   "faiss-cpu",
#   "requests",
#   "httpx",
#   "fastapi",
#   "uvicorn",
#   "pytesseract",
#   "slugify",
#   "pillow",
#   "requests",
#   "python-dotenv",
#   "python-slugify",
#   "google-genai",
# ]
# ///

import base64
import json
import mimetypes
import os
from io import BytesIO
from typing import Optional

import faiss
import httpx
import numpy as np
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from slugify import slugify
from google import genai
from google.genai import types
# Load env variables from .env
load_dotenv()

AIPROXY_URL = os.getenv("AIPROXY_URL")
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

import yaml
import logging

# Setup logger
logging.basicConfig(
    level=logging.INFO,  # Or DEBUG for more detail
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("log_info.log"),  # Logs to a file
        logging.StreamHandler()              # Also logs to console
    ]
)
logger = logging.getLogger("yaml_logger")

try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        logger.info("Loaded YAML config successfully")
        logger.debug(f"YAML contents: {config}")
except Exception as e:
    logger.error(f"Error loading YAML: {e}")

if not AIPROXY_URL or not AIPROXY_TOKEN:
    raise RuntimeError("Missing AIPROXY_URL or AIPROXY_TOKEN in environment variables.")

# === Load resources ===
print("🔹 Loading FAISS index...")
index = faiss.read_index("faiss_combined_index.idx")

print("🔹 Loading metadata...")
with open("embedding_combined_data.json", "r", encoding="utf-8") as f:
    embedding_data = json.load(f)

print("🔹 Loading embedding model...")
model = SentenceTransformer("./local_model")
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # === Utility ===
# def get_image_mimetype(base64_string):
#     image_data = base64.b64decode(base64_string)
#     try:
#         img = Image.open(BytesIO(image_data))
#         img_type = img.format.lower()  # e.g. 'jpeg', 'png'
#         mime_type = f'image/{img_type}'
#         extension = mimetypes.guess_extension(mime_type) or f".{img_type}"
#     except Exception as e:
#         print(f"Image type detection failed: {e}")
#         mime_type = 'application/octet-stream'
#         extension = '.bin'
#     return mime_type, extension, image_data
def extract_text_from_image(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        raise ValueError(f"OCR failed: {e}")


def get_image_mimetype(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        img = Image.open(BytesIO(image_data))
        img_type = img.format.lower()  # e.g. 'jpeg', 'png', 'webp'
        mime_type = f'image/{img_type}'
        extension = mimetypes.guess_extension(mime_type) or f".{img_type}"
        return mime_type, extension, image_data
    except Exception as e:
        raise ValueError("Could not determine the mimetype for your file — please ensure a valid image is provided.")

# Dummy placeholder for your Gemini integration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def get_image_description(image_bytes, mime_type):
    client = genai.Client(api_key=GEMINI_API_KEY)
    # Create a Part from bytes and mime type
    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type,
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            image_part,
            (
                "Provide a precise and structured description of the image. "
                "Focus only on the important content, such as error messages, warning banners, hash codes, URLs, input fields, buttons, scores, and any highlighted or emphasized elements. "
                "Ignore irrelevant decorations, ads, or backgrounds. "
                "If the image contains code or commands, transcribe it accurately. "
                "If there are error or status messages, report them exactly. "
                "If numeric values, scores, or hashes are present, output them cleanly. "
                "Format the description clearly using sections if needed (e.g., 'Error Message:', 'Hash:', 'Score:', 'URL:', 'Code:'). "
                "Do not make assumptions—describe only what is visibly present in the image."
            )
        ]
)
    return response.text

def query_faiss(query, top_k=5):
    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    
    D, I = index.search(query_emb, top_k)
    
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(embedding_data):
            results.append({
                "score": float(score),
                **embedding_data[idx]
            })

    final_results = []
    for res in results:
        base_entry = {
            "score": res["score"],
            "text_snippet": (res.get("combined_text") or res.get("chunk") or "")[:200]
        }

        if "topic_title" in res:
            base_entry["topic_title"] = res["topic_title"]
        
        if "root_post_number" in res:
            base_entry["root_post_number"] = res["root_post_number"]
        
        # Handle discourse-style URLs (list of URLs)
        if "url" in res and isinstance(res["url"], list):
            for url in res["url"]:
                entry = base_entry.copy()
                entry["url"] = url
                final_results.append(entry)
        
        # Handle TDS / notes style
        elif "original_url" in res:
            entry = base_entry.copy()
            entry["url"] = res["original_url"]
            final_results.append(entry)
        
        else:
            # fallback case if no URL
            final_results.append(base_entry)

    return final_results

def generate_llm_response(query, context_texts):
    context = "\n\n---\n\n".join(context_texts)
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on forum discussions and image description."
            },
            {
                "role": "user",
                "content": f"""<instructions>
You are a helpful assistant that answers questions based on forum discussions. 
First, think step-by-step about the relevant context. 
Provide your reasoning clearly, showing how each piece of the forum content contributes to your answer. 
Finally, give the answer in a concise and accurate way and also mention the image description given in query.
</instructions>
<context>
{context}
</context>
<question>
{query}
</question>
"""
            }
        ],
        "temperature": 0.3,
        "max_tokens": 400
    }

    response = httpx.post(AIPROXY_URL, headers=headers, json=payload, timeout=25.0)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return JSONResponse({"answer": f"test", "links": []})
        raise HTTPException(status_code=500, detail=f"AIPipe Error: {response.text}")



def answer(question, image):
    try:
        if image:
            print("got image")
            mime_type, extension, image_data = get_image_mimetype(image)
            print("Image open")
            image_response = get_image_description(image_data, mime_type)
            print("Got response")
            question = f"{question}\nImage description: {image_response}"
            print(image_response)

        results = query_faiss(question, top_k=10)

        print("FAISS results saved ")
        print(json.dumps(results, ensure_ascii=False, indent=4))

        url_text_list = []
        for item in results:
            url = item.get("url", "")
            text = item.get("text_snippet", "")
            url_text_list.append({
                "url": url,
                "text": text
            })
            
        context_texts = []
        for entry in url_text_list:
            url = entry.get("url", "")
            text = entry.get("text", "")
            combined = f"{url} {text}"
            context_texts.append(combined)

        # Get answer text from model
        response_text = generate_llm_response(question, context_texts)
        response_text = response_text or "No answer generated."

        # Save to a file
        # with open("llm_response.txt", "w", encoding="utf-8") as f:
        #     f.write(response_text)

        print("LLM response saved ")

        return JSONResponse({
            "answer": response_text,
            "links": url_text_list
        })
    except Exception as e:
        # Return a dict here; let the API endpoint wrap it in JSONResponse
        return {
            "answer": f"answer Error: {str(e)}",
            "links": []
        }


# === FastAPI app ===
app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "FastAPI is running!"}

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/api")
def api_answer(request: QueryRequest):
    try:
        return answer(request.question, request.image)
    except Exception as e:
        return JSONResponse({"answer": f"api Error: {str(e)}", "links": []})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0", port=7860)