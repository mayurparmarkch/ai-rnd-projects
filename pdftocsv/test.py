import hashlib
import re
import sys
import time
from pathlib import Path

import pdfplumber
from pdf2image import convert_from_path
import pytesseract

# --- Gemini API ---
import google.generativeai as genai

# =============== CONFIG ===============
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# TODO: Replace with your actual Gemini API key
GEMINI_API_KEY = "AIzaSyD-8SGcyPFvrrD6B4VDHtSTAkVpVyrRRp0"
# ======================================


# --- PDF Extraction ---
def extract_text_with_ocr(pdf_path: Path) -> str:
    """Extract text from image-based PDFs using OCR (Tesseract)."""
    text = ""
    pages = convert_from_path(str(pdf_path))
    for i, page in enumerate(pages, 1):
        extracted = pytesseract.image_to_string(page, lang="eng")
        text += f"--- OCR Page {i} Content ---\n{extracted}\n\n"
    return text

def looks_like_bad_text(text: str) -> bool:
    """Check if extracted text is too short or corrupted."""
    if len(text.strip()) < 50:  # very short
        return True
    non_alpha_ratio = len(re.findall(r"[^a-zA-Z0-9\s]", text)) / max(len(text), 1)
    return non_alpha_ratio > 0.5

def get_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF with caching and OCR fallback."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"🔴 ERROR: File not found -> {pdf_path}")

    pdf_bytes = pdf_path.read_bytes()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    cache_file = CACHE_DIR / f"{pdf_hash}.txt"

    if cache_file.exists():
        print("🧠 Found cached text, skipping extraction...")
        return cache_file.read_text("utf-8")

    print(f"📄 Extracting text from: {pdf_path.name}")
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            extracted = page.extract_text() or ""
            text += f"--- Page {i} Content ---\n{extracted}\n\n"

    if looks_like_bad_text(text):
        print("⚠️ Text looks incomplete or corrupted, switching to OCR...")
        text = extract_text_with_ocr(pdf_path)

    cache_file.write_text(text, "utf-8")
    print("✅ Text extracted and cached.")
    return text


# --- Gemini API for CSV ---
def generate_csv_from_text(pdf_text: str, user_prompt: str) -> str:
    """
    Generate CSV by sending text + prompt to Gemini.
    Uses caching for repeated requests.
    """
    request_hash = hashlib.sha256((pdf_text + user_prompt).encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"{request_hash}.csv"

    if cache_file.exists():
        print("🚀 Found matching request in cache. Returning result!")
        return cache_file.read_text("utf-8")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    You are a data analyst AI.
    Task: Based on the following PDF text, generate CSV data according to the request:

    Request: "{user_prompt}"

    Rules:
    - Output valid CSV only (RFC 4180).
    - Include a header row with descriptive column names.
    - No markdown, no explanations, only CSV.

    --- PDF Text ---
    {pdf_text}
    --- End PDF Text ---
    """

    print("🤖 Sending request to Gemini... Please wait.")
    spinner = "|/-\\"
    for i in range(20):
        sys.stdout.write(f"\r   Processing... {spinner[i % len(spinner)]}")
        sys.stdout.flush()
        time.sleep(0.1)

    response = model.generate_content(prompt, request_options={"timeout": 300})
    sys.stdout.write("\r" + " " * 20 + "\r")

    csv_output = response.text.strip()

    # Remove code block markers if AI added them
    if csv_output.startswith("```csv"):
        csv_output = csv_output[5:].strip()
    if csv_output.endswith("```"):
        csv_output = csv_output[:-3].strip()

    cache_file.write_text(csv_output, "utf-8")
    print("✅ CSV generated and cached.")
    return csv_output


def save_csv_file(csv_data: str, pdf_path: Path, user_prompt: str):
    """Save CSV to file with a descriptive name."""
    base_name = pdf_path.stem
    safe_prompt = re.sub(r"[^\w\s-]", "", user_prompt).strip().lower().replace(" ", "_")
    filename = f"{base_name}_{safe_prompt[:30]}.csv"
    Path(filename).write_text(csv_data, encoding="utf-8")
    print(f"💾 CSV saved as: {filename}")
    return filename


# --- Main ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pdf_to_csv.py <PDF_FILE> <PROMPT>")
        print('Example: python pdf_to_csv.py report.pdf "List all action items with owner and due date"')
        sys.exit(1)

    pdf_file = Path(sys.argv[1])
    user_prompt = " ".join(sys.argv[2:])  # dynamic prompt from CLI

    pdf_text = get_pdf_text(pdf_file)
    csv_data = generate_csv_from_text(pdf_text, user_prompt)
    save_csv_file(csv_data, pdf_file, user_prompt)

    print("\n--- CSV Preview ---\n")
    print("\n".join(csv_data.splitlines()[:10]))
    if len(csv_data.splitlines()) > 10:
        print("...")
