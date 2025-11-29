import os
import re
import hashlib
import uuid
from pathlib import Path
import pandas as pd
from difflib import SequenceMatcher
from io import StringIO
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import google.generativeai as genai
import json

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CACHE_DIR = Path(".cache")
OUTPUT_DIR = Path("output")
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

GEMINI_API_KEY = "AIzaSyB_7HUg5sC1fOMRAk4jS1HZHbichOvHqVE"
if not GEMINI_API_KEY:
    raise ValueError("🔴 ERROR: GEMINI_API_KEY not found.")

genai.configure(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def extract_text_with_ocr(pdf_path: Path) -> str:
    text = ""
    try:
        pages = convert_from_path(str(pdf_path), dpi=300)
        for i, page in enumerate(pages, 1):
            extracted = pytesseract.image_to_string(page, lang="eng")
            text += f"\n--- OCR Page {i} ---\n{extracted}\n"
    except Exception as e:
        return f"OCR failed: {e}"
    return text


def get_pdf_text(pdf_path: Path) -> str:
    pdf_bytes = pdf_path.read_bytes()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    cache_file = CACHE_DIR / f"{pdf_hash}.txt"
    if cache_file.exists():
        return cache_file.read_text("utf-8")

    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                extracted = page.extract_text() or ""
                text += f"\n--- Page {i} ---\n{extracted}\n"
    except Exception:
        text = extract_text_with_ocr(pdf_path)
    cache_file.write_text(text, "utf-8")
    return text


def calculate_similarity(a: str, b: str) -> float:
    return round(SequenceMatcher(None, str(a).lower().strip(), str(b).lower().strip()).ratio() * 100, 2)


# ---------------------------------------------------------
# Flask App
# ---------------------------------------------------------
app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return jsonify({"message": "PDF to FotonVR Mapper API v3"})


# 🧠 Step 1: Extract structured data
@app.route("/extract", methods=["POST"])
def extract_pdf_data():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Missing file"}), 400

        pdf_file = request.files["file"]
        pdf_bytes = pdf_file.read()  # read bytes first for hashing
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

        pdf_path = CACHE_DIR / f"{pdf_hash}.pdf"
        pdf_path.write_bytes(pdf_bytes)

        # --- Check if cached structured JSON already exists ---
        cache_json_path = CACHE_DIR / f"{pdf_hash}.json"
        if cache_json_path.exists():
            print("⚡ Using cached structured data")
            cached_data = json.loads(cache_json_path.read_text("utf-8"))
            return jsonify({"message": "Loaded from cache", "data": cached_data})

        # --- Otherwise process fresh ---
        print("🧠 Processing new PDF with Gemini...")
        pdf_text = get_pdf_text(pdf_path)

        # ✅ Handle large text by splitting into safe chunks
        MAX_CHARS = 180000  # increase from 10k to handle large docs
        chunks = [pdf_text[i:i + MAX_CHARS] for i in range(0, len(pdf_text), MAX_CHARS)]
        print(f"📄 PDF split into {len(chunks)} chunks")

        model = genai.GenerativeModel("gemini-2.5-flash")
        all_results = []

        for idx, chunk in enumerate(chunks, 1):
            print(f"🧩 Processing chunk {idx}/{len(chunks)} ({len(chunk)} chars)")
            prompt = f"""
Extract structured educational data from the following text.
Return ONLY JSON array of objects like:
[
  {{"title": "...", "chapter": "...", "topic": "...", "description": "..."}}
]
Chunk {idx}:
{chunk}
"""
            response = model.generate_content(prompt)
            text_output = ""
            if response.candidates:
                for c in response.candidates:
                    if c.content and c.content.parts:
                        for p in c.content.parts:
                            if hasattr(p, "text"):
                                text_output += p.text

            clean = re.sub(r"^```json|```$", "", text_output.strip()).strip()
            try:
                data = json.loads(clean)
                if isinstance(data, list):
                    all_results.extend(data)
            except Exception as e:
                print(f"⚠️ Chunk {idx} failed to parse JSON:", e)
                continue

        if not all_results:
            return jsonify({"error": "Gemini returned no valid structured data"}), 500

        # --- Save structured data to cache ---
        cache_json_path.write_text(json.dumps(all_results, indent=2), "utf-8")

        return jsonify({"message": "Extracted successfully", "data": all_results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



# 🧩 Step 2: Map extracted data with uploaded CSV (AI-based)
@app.route("/map-extracted", methods=["POST"])
def map_extracted_data():
    try:
        if "csv" not in request.files or "data" not in request.form:
            return jsonify({"error": "Missing CSV or extracted data"}), 400

        # --- Load reference CSV ---
        ref_file = request.files["csv"]
        ref_path = CACHE_DIR / f"ref_{uuid.uuid4().hex}.csv"
        ref_file.save(ref_path)
        ref_df = pd.read_csv(ref_path)

        uuid_col = next((c for c in ref_df.columns if "uuid" in c.lower() or "id" == c.lower()), None)
        if not uuid_col:
            return jsonify({"error": "CSV must contain a UUID or ID column"}), 400

        extracted_data = json.loads(request.form["data"])

        # --- Convert CSV into JSON list for context ---
        csv_data = ref_df.to_dict(orient="records")

        # --- Prepare AI prompt ---
        prompt = f"""
You are an intelligent data mapping assistant.

You are given two datasets:
1. **Extracted Data** — from a PDF (list of objects with title, description, etc.)
2. **Reference Data** — from a CSV (list of objects with a UUID column and other metadata)
    
Your goal:
- For each extracted object, find the **best matching reference record**.
- Match primarily based on semantic meaning of title/description.
- If no good match is found, set `matched_uuid` to null.
- Return **ONLY** a valid JSON list of objects, where each item includes:
  - All original extracted fields
  - `matched_uuid`: the UUID from reference data (or null)
  - `match_reason`: a short reason why it matched
  - `match_confidence`: a number from 0–100 estimating confidence

Here is the reference dataset (shortened form):
{json.dumps(csv_data[:50], indent=2)}  # limit to 50 rows to fit safely

Now map the following extracted data:
{json.dumps(extracted_data, indent=2)}
"""

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        # --- Extract AI output safely ---
        text_output = ""
        if response.candidates:
            for c in response.candidates:
                if c.content and c.content.parts:
                    for p in c.content.parts:
                        if hasattr(p, "text"):
                            text_output += p.text

        clean = re.sub(r"^```json|```$", "", text_output.strip()).strip()
        try:
            mapped_data = json.loads(clean)
        except Exception as e:
            print("⚠️ AI mapping JSON parse failed:", e)
            return jsonify({"error": "AI mapping failed, invalid JSON returned", "raw_output": text_output}), 500

        return jsonify({"message": "AI Mapping complete", "data": mapped_data})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Running on http://localhost:5000")
    app.run(debug=True, port=5000)















