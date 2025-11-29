import os
import re
import hashlib
import uuid
import json
from pathlib import Path
import pandas as pd
from difflib import SequenceMatcher
from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import google.generativeai as genai

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CACHE_DIR = Path(".cache")
OUTPUT_DIR = Path("output")
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

GEMINI_API_KEY = "AIzaSyB_7HUg5sC1fOMRAk4jS1HZHbichOvHqVE"
genai.configure(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def extract_text_with_ocr(pdf_path: Path) -> str:
    """Fallback OCR extraction if pdfplumber fails."""
    text = ""
    try:
        pages = convert_from_path(str(pdf_path), dpi=300)
        for i, page in enumerate(pages, 1):
            text += f"\n--- OCR Page {i} ---\n" + pytesseract.image_to_string(page, lang="eng") + "\n"
    except Exception as e:
        text = f"OCR failed: {e}"
    return text


def get_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF, cached by SHA256."""
    pdf_bytes = pdf_path.read_bytes()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    cache_file = CACHE_DIR / f"{pdf_hash}.txt"
    if cache_file.exists():
        return cache_file.read_text("utf-8")

    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text += f"\n--- Page {i} ---\n" + (page.extract_text() or "") + "\n"
    except Exception:
        text = extract_text_with_ocr(pdf_path)

    cache_file.write_text(text, "utf-8")
    return text


def safe_json_loads(raw: str):
    """Try to clean and parse slightly malformed JSON."""
    clean = raw.strip().strip("`").strip()
    clean = re.sub(r"^```json|```$", "", clean, flags=re.MULTILINE).strip()
    clean = re.sub(r",\s*}", "}", clean)
    clean = re.sub(r",\s*]", "]", clean)
    clean = clean.replace("\n", " ")
    try:
        return json.loads(clean)
    except Exception:
        # Try to isolate the first valid JSON list
        m = re.search(r"\[.*\]", clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


# ---------------------------------------------------------
# Flask App
# ---------------------------------------------------------
app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return jsonify({"message": "PDF to FotonVR Mapper API v4"})


# 🧠 Step 1: Extract structured data
@app.route("/extract", methods=["POST"])
def extract_pdf_data():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Missing file"}), 400

        pdf_file = request.files["file"]
        pdf_bytes = pdf_file.read()
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
        pdf_path = CACHE_DIR / f"{pdf_hash}.pdf"
        pdf_path.write_bytes(pdf_bytes)

        cache_json_path = CACHE_DIR / f"{pdf_hash}.json"
        if cache_json_path.exists():
            cached = json.loads(cache_json_path.read_text("utf-8"))
            return jsonify({"message": "Loaded from cache", "data": cached})

        pdf_text = get_pdf_text(pdf_path)
        chunks = [pdf_text[i:i + 150000] for i in range(0, len(pdf_text), 150000)]

        model = genai.GenerativeModel("gemini-2.5-flash")
        all_results = []
        for i, chunk in enumerate(chunks, 1):
            prompt = f"""
Extract structured data from this textbook text as JSON array only.
Each object must contain:
- chapter
- title
- topic
- description
NO extra commentary, NO markdown. Only JSON array.

Chunk {i}:
{chunk}
"""
            response = model.generate_content(prompt)
            text_output = "".join(
                p.text for c in response.candidates for p in getattr(c.content, "parts", [])
                if hasattr(p, "text")
            )
            data = safe_json_loads(text_output)
            if data:
                all_results.extend(data)

        if not all_results:
            return jsonify({"error": "No valid structured data returned"}), 500

        cache_json_path.write_text(json.dumps(all_results, indent=2))
        return jsonify({"message": "Extracted successfully", "data": all_results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 🧩 Step 2: Map extracted data to CSV
@app.route("/map-extracted", methods=["POST"])
def map_extracted_data():
    try:
        if "csv" not in request.files or "data" not in request.form:
            return jsonify({"error": "Missing CSV or extracted data"}), 400

        csv_file = request.files["csv"]
        ref_path = CACHE_DIR / f"ref_{uuid.uuid4().hex}.csv"
        csv_file.save(ref_path)
        ref_df = pd.read_csv(ref_path)

        extracted_data = json.loads(request.form["data"])
        csv_data = ref_df.to_dict(orient="records")

        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
Map the following PDF-extracted data with this CSV dataset by meaning.

Return JSON list where each object includes:
- all fields from extracted data
- matched_uuid (UUID from CSV if found)
- match_confidence (0–100)
- match_reason (why it matched)
- why_lesser (missing concepts)
- pdf_summary (only once, summarizing PDF in 2–3 sentences at the end)

Reference CSV (trimmed):
{json.dumps(csv_data[:40], indent=2)}

Extracted data:
{json.dumps(extracted_data[:60], indent=2)}
"""

        response = model.generate_content(prompt)
        text_output = "".join(
            p.text for c in response.candidates for p in getattr(c.content, "parts", [])
            if hasattr(p, "text")
        )

        mapped_data = safe_json_loads(text_output)
        if not mapped_data:
            return jsonify({"error": "AI mapping failed, invalid JSON returned", "raw_output": text_output}), 500

        # ✅ Auto-detect summary & rename properly
        for d in mapped_data:
            if "summary" in d and "pdf_summary" not in d:
                d["pdf_summary"] = d.pop("summary")
            elif "overview" in d and "pdf_summary" not in d:
                d["pdf_summary"] = d.pop("overview")

        return jsonify({"message": "AI Mapping complete", "data": mapped_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     print("🚀 Running on http://localhost:5000")
#     app.run(debug=True, port=5000)




if __name__ == "__main__":
    import sys

    host = "0.0.0.0"   # default → accessible on LAN
    port = 5000        # default port

    # ✅ Allow: python index2.py --host 192.168.1.145 --port 6000
    if "--host" in sys.argv:
        host = sys.argv[sys.argv.index("--host") + 1]

    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])

    print(f"🚀 Running on http://{host}:{port}")
    app.run(debug=True, host=host, port=port)
