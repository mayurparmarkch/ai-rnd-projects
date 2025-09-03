import fitz
import json
import requests

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF file.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A single string containing all the text from the PDF.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except FileNotFoundError:
        return f"Error: The file at {pdf_path} was not found."
    except Exception as e:
        return f"An error occurred: {e}"

def generate_syllabus_from_text(text: str, api_key: str) -> str:
    """
    Generates a syllabus by sending the text to the Gemini API and
    parsing the JSON response.
    """
    try:
        system_prompt = (
            "You are a helpful assistant that generates a syllabus from provided text. "
            "Your output must be in a JSON format. The JSON should be an array of objects, "
            "where each object has a 'topic' and a 'subtopics' property. "
            "The 'subtopics' property should be an array of strings."
        )
        user_query = f"Generate a syllabus from the following text:\n\n{text}"

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": user_query}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "topic": {"type": "STRING"},
                            "subtopics": {
                                "type": "ARRAY",
                                "items": {"type": "STRING"}
                            }
                        }
                    }
                }
            }
        }

        # Send request to Gemini API
        response = requests.post(api_url, json=payload)
        response.raise_for_status()

        # Parse JSON from API
        data = response.json()
        if "candidates" in data and len(data["candidates"]) > 0:
            syllabus_data = json.loads(data["candidates"][0]["content"])
        else:
            return "Error: No valid syllabus generated from API."

        # Format syllabus for printing
        syllabus_lines = ["Generated Syllabus:\n"]
        for item in syllabus_data:
            syllabus_lines.append(f"**{item['topic']}**")
            for subtopic in item['subtopics']:
                syllabus_lines.append(f"  * {subtopic}")
            syllabus_lines.append("-" * 30)

        return "\n".join(syllabus_lines)

    except requests.exceptions.RequestException as e:
        return f"Error: Failed to call API - {e}"
    except json.JSONDecodeError:
        return "Error: Failed to parse JSON response from API."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python generate_syllabus.py <PDF_PATH> <API_KEY>")
        sys.exit(1)

    pdf_file_path = sys.argv[1]
    api_key = sys.argv[2]

    document_text = extract_text_from_pdf(pdf_file_path)
    if not document_text.startswith("Error"):
        syllabus = generate_syllabus_from_text(document_text, api_key)
        print(syllabus)
    else:
        print(document_text)
