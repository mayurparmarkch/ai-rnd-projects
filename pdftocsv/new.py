import os
import sys
import re
import pdfplumber
import google.generativeai as genai


# --- Step 1: Extract text from PDF ---
def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


# --- Step 2: Generate CSV with Gemini ---
def generate_csv_from_pdf(pdf_path, user_prompt):
    api_key = "AIzaSyD-8SGcyPFvrrD6B4VDHtSTAkVpVyrRRp0"  # replace with env var in prod

    if not api_key:
        raise ValueError("Please set GEMINI_API_KEY in your environment.")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    text = extract_pdf_text(pdf_path)

    prompt = f"""
    You are given raw text extracted from a PDF:

    {text}

    Instruction from user:
    {user_prompt}

    Please:
    - Extract data according to the instruction
    - Return output **only in valid CSV format** with a header row.
    - Do not include explanations, only CSV.
    """

    response = model.generate_content(prompt)
    csv_output = response.text.strip()
    return csv_output


# --- Step 3: Extract file name from instruction ---
def get_csv_filename_from_prompt(prompt):
    # Try to find something like "invoice CSV" or "customer CSV"
    match = re.search(r'(\w+)\s+CSV', prompt, re.IGNORECASE)
    if match:
        base_name = match.group(1).lower()
    else:
        base_name = "output"

    return f"{base_name}.csv"


# --- Step 4: Run dynamically ---
if __name__ == "__main__":
    pdf_file = "test.pdf"

    # Option A: CLI argument
    if len(sys.argv) > 1:
        instruction = " ".join(sys.argv[1:])
    else:
        # Option B: Interactive input
        instruction = input("Enter your instruction (e.g., Generate invoice CSV with Title and Price): ")

    csv_data = generate_csv_from_pdf(pdf_file, instruction)

    # Generate CSV filename dynamically
    csv_filename = get_csv_filename_from_prompt(instruction)

    # Save file
    with open(csv_filename, "w", encoding="utf-8") as f:
        f.write(csv_data)

    print(f"✅ CSV file generated: {csv_filename}\n")
    print(csv_data)
