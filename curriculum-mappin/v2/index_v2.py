from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pdfplumber
import uuid
import json
import google.generativeai as genai
from pathlib import Path
import re
import os

app = Flask(__name__)
CORS(app)

# 🔑 Configure Gemini
genai.configure(api_key="YOUR_GEMINI_API_KEY")  # Replace with your real key
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# ✅ Extract text from PDF
def get_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


# ✅ STEP 1: Extract structured data from PDF
@app.route("/extract", methods=["POST"])
def extract_pdf_data():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No PDF file uploaded"}), 400

        pdf_file = request.files["file"]
        pdf_path = CACHE_DIR / f"{uuid.uuid4().hex}.pdf"
        pdf_file.save(pdf_path)

        pdf_text = get_pdf_text(pdf_path)
        if not pdf_text.strip():
            return jsonify({"error": "PDF has no readable text"}), 400

        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""
Extract structured educational data from this PDF text.
Return only a JSON array of objects with keys:
["title", "chapter", "topic", "description"]

PDF Content:
{pdf_text[:15000]}
"""

        response = model.generate_content(prompt)
        result_text = ""

        # 🧩 Safely extract Gemini text parts
        try:
            if response.candidates:
                for c in response.candidates:
                    if c.content and c.content.parts:
                        for p in c.content.parts:
                            if hasattr(p, "text") and p.text:
                                result_text += p.text
        except Exception:
            pass

        # 🧹 Clean & parse JSON
        clean = result_text.strip()
        clean = re.sub(r"^```json|```$", "", clean, flags=re.MULTILINE).strip()

        try:
            data = json.loads(clean)
            if not isinstance(data, list):
                data = [{"title": "Parsing Error", "chapter": "N/A", "topic": "N/A", "description": clean}]
        except Exception:
            # If Gemini output isn't valid JSON, wrap it for debugging
            data = [{"title": "Parsing Error", "chapter": "N/A", "topic": "N/A", "description": clean or "No data"}]

        return jsonify({"data": data})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ✅ STEP 2: Map extracted data with CSV (append uuid or null)
@app.route("/process", methods=["POST"])
def process_mapping():
    try:
        if "pdf_data" not in request.form or "csv_file" not in request.files:
            return jsonify({"error": "Missing data or CSV file"}), 400

        pdf_data = json.loads(request.form["pdf_data"])
        csv_file = request.files["csv_file"]

        df_csv = pd.read_csv(csv_file)
        csv_records = df_csv.to_dict(orient="records")

        model = genai.GenerativeModel("gemini-2.0-flash")
        results = []

        for item in pdf_data:
            prompt = f"""
Match the following extracted item with one of these CSV entries.

Item:
{json.dumps(item, indent=2)}

CSV entries:
{json.dumps(csv_records[:40], indent=2)}

Return only JSON:
{{ "matched_uuid": "<uuid from CSV or null>" }}
"""

            response = model.generate_content(prompt)
            res_text = ""
            try:
                if response.candidates:
                    for c in response.candidates:
                        if c.content and c.content.parts:
                            for p in c.content.parts:
                                if hasattr(p, "text"):
                                    res_text += p.text
            except Exception:
                pass

            clean = re.sub(r"^```json|```$", "", res_text.strip()).strip()
            try:
                matched = json.loads(clean)
                item["uuid"] = matched.get("matched_uuid", None)
            except:
                item["uuid"] = None

            results.append(item)

        return jsonify({"mapped_data": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
