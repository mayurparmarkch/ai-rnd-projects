import os
import re
import hashlib
from pathlib import Path

# Third-party libraries
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# --- Gemini API ---
import google.generativeai as genai

# Create necessary directories
CACHE_DIR = Path(".cache")
OUTPUT_DIR = Path("output")
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Securely configure the Gemini API key
GEMINI_API_KEY = "AIzaSyD-8SGcyPFvrrD6B4VDHtSTAkVpVyrRRp0";

if not GEMINI_API_KEY:
    raise ValueError("🔴 ERROR: GEMINI_API_KEY not found. Please set it in a .env file.")
genai.configure(api_key=GEMINI_API_KEY)
# =====================================================


# --- PDF Extraction Logic ---
def extract_text_with_ocr(pdf_path: Path) -> str:
    """Extract text from image-based PDFs using OCR (Tesseract)."""
    print("⏳ Running OCR fallback...")
    text = ""
    try:
        pages = convert_from_path(str(pdf_path))
        for i, page in enumerate(pages, 1):
            extracted = pytesseract.image_to_string(page, lang="eng")
            text += f"--- OCR Page {i} Content ---\n{extracted}\n\n"
        print("✅ OCR finished.")
    except Exception as e:
        print(f"🔴 OCR Error: {e}")
        return f"OCR processing failed: {e}"
    return text

def looks_like_bad_text(text: str) -> bool:
    """Check if extracted text is too short or corrupted."""
    if len(text.strip()) < 50:
        return True
    # Check for a high ratio of non-alphanumeric characters, suggesting gibberish
    non_alpha_ratio = len(re.findall(r"[^a-zA-Z0-9\s]", text)) / max(len(text), 1)
    return non_alpha_ratio > 0.5

def get_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF with caching and OCR fallback."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"🔴 ERROR: File not found -> {pdf_path}")

    # Create a hash of the file content to use for caching
    pdf_bytes = pdf_path.read_bytes()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    cache_file = CACHE_DIR / f"{pdf_hash}.txt"

    if cache_file.exists():
        print(f"✅ Found cached text for {pdf_path.name}")
        return cache_file.read_text("utf-8")

    print(f"📄 Extracting text from {pdf_path.name}...")
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                extracted = page.extract_text() or ""
                text += f"--- Page {i} Content ---\n{extracted}\n\n"
    except Exception as e:
        print(f"🔴 pdfplumber Error: {e}. Attempting OCR as a fallback.")
        text = extract_text_with_ocr(pdf_path) # Fallback to OCR on pdfplumber error too
        cache_file.write_text(text, "utf-8")
        return text

    if looks_like_bad_text(text):
        print("⚠️ Extracted text looks corrupted. Trying OCR.")
        text = extract_text_with_ocr(pdf_path)

    cache_file.write_text(text, "utf-8")
    print("✅ Text extraction complete.")
    return text


# --- Gemini API for CSV Generation ---
def generate_csv_from_text(pdf_text: str, user_prompt: str) -> str:
    """Generates CSV data from text using the Gemini API."""
    # Create a hash for the combined text and prompt to cache the AI response
    request_hash = hashlib.sha256((pdf_text + user_prompt).encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"{request_hash}.csv"

    if cache_file.exists():
        print("✅ Found cached CSV response.")
        return cache_file.read_text("utf-8")

    print("🧠 Calling Gemini API to generate CSV...")
    model = genai.GenerativeModel("gemini-2.5-pro")
    prompt = f"""
    You are an expert data analyst AI. Your task is to extract information from the provided PDF text and format it as valid CSV data based on the user's request.

    **User Request:** "{user_prompt}"

    **Rules:**
    1.  **CSV Only:** Your entire output must be in valid CSV format (RFC 4180).
    2.  **Header Row:** The first line must be a descriptive header row.
    3.  **No Extra Text:** Do not include any explanations, notes, or markdown formatting like ```csv ... ``` in your response. Only output the raw CSV data.

    **--- PDF Text ---**
    {pdf_text}
    **--- End PDF Text ---**
    """
    try:
        response = model.generate_content(prompt, request_options={"timeout": 300})
        csv_output = response.text.strip()
        
        # Clean up potential markdown fences just in case
        if csv_output.startswith("```csv"):
            csv_output = csv_output[5:].strip()
        if csv_output.endswith("```"):
            csv_output = csv_output[:-3].strip()

        # Final validation to ensure it's not empty
        if not csv_output or len(csv_output.splitlines()) < 1:
             raise ValueError("Generated CSV is empty.")

        cache_file.write_text(csv_output, "utf-8")
        print("✅ Gemini API call successful.")
        return csv_output
    except Exception as e:
        print(f"🔴 Gemini API Error: {e}")
        raise ValueError(f"Failed to generate content from AI model: {e}")


def save_csv_file(csv_data: str, pdf_path: Path, user_prompt: str) -> str:
    """Saves the CSV data to a file in the output directory."""
    base_name = pdf_path.stem
    # Create a safe filename from the user prompt
    safe_prompt = re.sub(r"[^\w\s-]", "", user_prompt).strip().lower().replace(" ", "_")
    filename = f"{base_name}_{safe_prompt[:30]}.csv"
    output_path = OUTPUT_DIR / filename
    output_path.write_text(csv_data, encoding="utf-8")
    print(f"💾 CSV file saved as: {output_path}")
    return filename


# --- Flask API ---
app = Flask(__name__)
CORS(app)  # This enables Cross-Origin Resource Sharing

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the PDF to CSV API! Use the /process endpoint."})

@app.route("/process", methods=["POST"])
def process_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    if "prompt" not in request.form or not request.form["prompt"]:
        return jsonify({"error": "Missing 'prompt' in the request form"}), 400

    file = request.files["file"]
    prompt = request.form["prompt"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Sanitize filename to prevent directory traversal issues
    safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', file.filename)
    pdf_path = CACHE_DIR / safe_filename
    file.save(pdf_path)

    try:
        # Step 1: Extract text from PDF
        pdf_text = get_pdf_text(pdf_path)
        if not pdf_text.strip():
            return jsonify({"error": "Could not extract any text from the PDF. It might be empty or corrupted."}), 500

        # Step 2: Generate CSV using Gemini
        csv_data = generate_csv_from_text(pdf_text, prompt)
        
        if len(csv_data.splitlines()) < 2:
            return jsonify({"error": "AI failed to generate valid CSV. The PDF content might not contain the data you requested."}), 500
            
        # Step 3: Save the CSV file
        filename = save_csv_file(csv_data, pdf_path, prompt)

        # Step 4: Return a successful response
        return jsonify({
            "message": "CSV generated successfully!",
            "csv_filename": filename,
            "csv_preview": "\n".join(csv_data.splitlines()[:10]) # Send top 10 lines for preview
        })
    except FileNotFoundError as e:
        return jsonify({"error": f"File operation failed: {str(e)}"}), 500
    except ValueError as e: # Catches Gemini API errors
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"🔴 An unexpected error occurred: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route("/download/<filename>")
def download_file(filename):
    """Serves files from the output directory for download."""
    # Security: Ensure filename is safe and doesn't allow directory traversal
    if ".." in filename or filename.startswith("/"):
        return jsonify({"error": "Invalid filename"}), 400
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)