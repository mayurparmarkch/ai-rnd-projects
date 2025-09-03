import os
import pdfplumber
import google.generativeai as genai

# --- Step 1: Extract text from PDF ---
text = ""
with pdfplumber.open("test.pdf") as pdf:
    for page in pdf.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

# --- Step 2: Use Gemini to structure text ---
api_key = "AIzaSyD-8SGcasdasdyPFvrrD6B4VDHtSTAkVpVyrRRp0"  # Set your API key as an environment variable
if not api_key:
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

prompt = f"""
Here is raw text extracted from a PDF:

{text}

Please:
- Reconstruct missing formatting
- Maintain structure (headings, subheadings, paragraphs, bullet points, tables if possible)
- Return the output in well-structured Markdown format.
"""

try:
    response = model.generate_content(prompt)
    # Print the whole response to inspect its structure
    print(response)
    # Try to access the text content
    if hasattr(response, "text"):
        print(response.text)
    elif hasattr(response, "candidates"):
        print(response.candidates[0].content.parts[0].text)
    else:
        print("Unexpected response format.")
except Exception as e:
    print(f"Error: {e}")