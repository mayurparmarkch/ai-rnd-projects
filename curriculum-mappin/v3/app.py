from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
from flask_cors import CORS
from pathlib import Path
import hashlib, json, uuid, re
import pandas as pd
import pdfplumber
import google.generativeai as genai

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
app = Flask(__name__, template_folder="templates")
CORS(app)

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
genai.configure(api_key="AIzaSyB_7HUg5sC1fOMRAk4jS1HZHbichOvHqVE")  # 👈 replace this

# ---------------------------------------------------------
# Helper: Extract text from PDF
# ---------------------------------------------------------
def get_pdf_text(pdf_path: Path) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            extracted = page.extract_text() or ""
            text += f"\n--- Page {i} ---\n{extracted}\n"
    return text

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route("/")
def home():
    return redirect(url_for("extract_page"))

@app.route("/extract-page")
def extract_page():
    return render_template("extract.html")

@app.route("/map-page")
def map_page():
    return render_template("map.html")

# ---------------------------------------------------------
# Step 1 - Extract data from PDF
# ---------------------------------------------------------
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
            return jsonify({"message": "Loaded from cache", "hash": pdf_hash, "data": cached})

        # Extract text and send to Gemini
        model = genai.GenerativeModel("gemini-2.5-flash")
        pdf_text = get_pdf_text(pdf_path)

        prompt = f"""
Extract structured educational data from this PDF text.
Return JSON list in this exact format:
[{{"chapter": "...", "title": "...", "topic": "...", "description": "..."}}]

Text:
{pdf_text[:180000]}
"""
        response = model.generate_content(prompt)

        text_output = ""
        for c in response.candidates:
            for p in c.content.parts:
                if hasattr(p, "text"):
                    text_output += p.text

        clean = re.sub(r"^```json|```$", "", text_output.strip()).strip()

        try:
            data = json.loads(clean)
        except Exception:
            data = [{"chapter": "-", "title": "-", "topic": "-", "description": "Parsing error"}]

        cache_json_path.write_text(json.dumps(data, indent=2), "utf-8")
        return jsonify({"message": "Extracted successfully", "hash": pdf_hash, "data": data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------
# Step 2 - Map with CSV
# ---------------------------------------------------------
@app.route("/map-extracted", methods=["POST"])
def map_extracted():
    try:
        if "csv" not in request.files or "hash" not in request.form:
            return jsonify({"error": "Missing file or hash"}), 400

        pdf_hash = request.form["hash"]
        cache_json_path = CACHE_DIR / f"{pdf_hash}.json"
        if not cache_json_path.exists():
            return jsonify({"error": "Extracted data not found"}), 404

        extracted_data = json.loads(cache_json_path.read_text("utf-8"))

        ref_file = request.files["csv"]
        ref_path = CACHE_DIR / f"ref_{uuid.uuid4().hex}.csv"
        ref_file.save(ref_path)
        ref_df = pd.read_csv(ref_path)

        uuid_col = next((c for c in ref_df.columns if "uuid" in c.lower() or "id" in c.lower()), None)
        if not uuid_col:
            return jsonify({"error": "CSV must contain a UUID column"}), 400

        csv_data = ref_df.to_dict(orient="records")

        model = genai.GenerativeModel("gemini-2.5-flash")

        # Create mapping prompt including confidence + missing info + summary
        prompt = f"""
You are analyzing extracted PDF data and a CSV reference file.

Task:
1. Match extracted PDF topics with CSV topics based on meaning and similarity.
2. For each extracted item, output:
   - "chapter"
   - "title"
   - "topic"
   - "description"
   - "matched_uuid" (UUID from CSV or null)
   - "confidence_score" (as percentage how well it matched)
   - "why_it_lesser" (list topics or points missing from the CSV side)
3. Add at the end ONE extra row (key: "pdf_summary") that contains a **very short summary** (1–2 sentences) of what this PDF is about.

Return JSON list in this format:
[
  {{
    "chapter": "...",
    "title": "...",
    "topic": "...",
    "description": "...",
    "matched_uuid": "...",
    "confidence_score": "85%",
    "why_it_lesser": "missing topics: ...",
    "pdf_summary": null
  }},
  {{
    "pdf_summary": "This PDF explains thermodynamics laws briefly..."
  }}
]

Reference CSV (truncated to 50 rows):
{json.dumps(csv_data[:50], indent=2)}

Extracted PDF data:
{json.dumps(extracted_data, indent=2)}
"""

        response = model.generate_content(prompt)
        text_output = ""
        for c in response.candidates:
            for p in c.content.parts:
                if hasattr(p, "text"):
                    text_output += p.text

        clean = re.sub(r"^```json|```$", "", text_output.strip()).strip()
        mapped_data = json.loads(clean)

        return jsonify({"message": "Mapping complete", "data": mapped_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
