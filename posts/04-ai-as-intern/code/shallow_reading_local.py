# Created: 2026-03-17 12:20
# Purpose: Read a folder of archival images with the OpenAI API and save one structured JSON file per image.
# Create a subdirectory called "images" and put your image files in there. Then run this script. The output JSON files will be saved to the "output_json" folder.
# To run this script, you need to have Python installed, along with the OpenAI Python package. You also need an OpenAI API key.


from pathlib import Path
import base64
import json
import mimetypes
import logging
import sys
from datetime import datetime

# -------- CONFIG --------
API_KEY = ""  # Optional fallback. If api_key.txt exists in this script's folder, that file is used first.
INPUT_DIR = Path("./images")
OUTPUT_DIR = Path("./output_json")
MODEL = "gpt-5.2"
LOG_FILE = "image_to_json_errors.log"
# ------------------------

PROMPT = """
You are reading one historical document image.

Do shallow reading only.
Do not overinterpret.
Do not invent context that is not visible on the page.

Return structured JSON with:
- file_summary: brief plain summary of what is on the page
- transcription: best-effort transcription of visible text
- named_entities:
  - people
  - places
  - organizations
  - dates
  - addresses
  - money
- categories:
  - medical
  - legal
  - property
  - labor
  - family
  - government
  - education
  - religion
  - military
  - other
- key_facts: short list of factual points visible on the page
- language
- document_type
- confidence_notes: short note about uncertainty, legibility, damage, handwriting, etc.

Rules:
- If something is unclear, say so.
- If a category is not present, use an empty list or false.
- Keep extraction literal and conservative.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "file_summary": {"type": "string"},
        "transcription": {"type": "string"},
        "named_entities": {
            "type": "object",
            "properties": {
                "people": {"type": "array", "items": {"type": "string"}},
                "places": {"type": "array", "items": {"type": "string"}},
                "organizations": {"type": "array", "items": {"type": "string"}},
                "dates": {"type": "array", "items": {"type": "string"}},
                "addresses": {"type": "array", "items": {"type": "string"}},
                "money": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["people", "places", "organizations", "dates", "addresses", "money"],
            "additionalProperties": False
        },
        "categories": {
            "type": "object",
            "properties": {
                "medical": {"type": "boolean"},
                "legal": {"type": "boolean"},
                "property": {"type": "boolean"},
                "labor": {"type": "boolean"},
                "family": {"type": "boolean"},
                "government": {"type": "boolean"},
                "education": {"type": "boolean"},
                "religion": {"type": "boolean"},
                "military": {"type": "boolean"},
                "other": {"type": "array", "items": {"type": "string"}}
            },
            "required": [
                "medical", "legal", "property", "labor", "family",
                "government", "education", "religion", "military", "other"
            ],
            "additionalProperties": False
        },
        "key_facts": {"type": "array", "items": {"type": "string"}},
        "language": {"type": "string"},
        "document_type": {"type": "string"},
        "confidence_notes": {"type": "string"}
    },
    "required": [
        "file_summary",
        "transcription",
        "named_entities",
        "categories",
        "key_facts",
        "language",
        "document_type",
        "confidence_notes"
    ],
    "additionalProperties": False
}


def setup_logging(script_dir: Path) -> Path:
    log_path = script_dir / LOG_FILE
    logging.basicConfig(
        filename=str(log_path),
        level=logging.ERROR,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return log_path


def import_openai_or_explain():
    try:
        from openai import OpenAI
        return OpenAI
    except ImportError:
        print("""
ERROR: The 'openai' package is not installed.

To fix this, create a virtual environment and install the package:

Mac/Linux:
  python3 -m venv shallow_env
  source shallow_env/bin/activate
  python -m pip install openai

Windows (PowerShell):
  py -m venv shallow_env
  shallow_env\\Scripts\\Activate.ps1
  python -m pip install openai

If activation fails on Windows, run:
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Then run the script again.
""")
        sys.exit(1)

def resolve_api_key(script_dir: Path) -> str:
    api_key_file = script_dir / "api_key.txt"

    if api_key_file.exists():
        key = api_key_file.read_text(encoding="utf-8").strip()
        if key:
            print(f"Using API key from: {api_key_file}")
            return key
        else:
            print(f"Found {api_key_file}, but it is empty.")

    if API_KEY.strip():
        print("Using API key from CONFIG section.")
        return API_KEY.strip()

    print("\nERROR: No API key found.\n")
    print("What to do:\n")

    print(
        "An API is a connection between your software and the cloud.\n"
        "The API key is like a password that allows this script to use the OpenAI API.\n"
    )

    print("To learn more about API keys and how to get one:")
    print("https://www.merge.dev/blog/chatgpt-api-key\n")

    print("1. Put your API key into a file named api_key.txt in the same folder as this script")
    print("   OR")
    print("2. Paste your API key into the API_KEY variable in the CONFIG section.\n")

    print("Example api_key.txt contents:")
    print("  sk-...\n")

    sys.exit(1)


def image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None:
        mime_type = "application/octet-stream"
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def get_response_text(response) -> str:
    # Preferred when supported by the SDK
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    # Fallback for older/different SDK response structures
    try:
        return response.output[0].content[0].text
    except Exception as e:
        raise RuntimeError(f"Could not extract text from API response: {e}")


def main():
    script_dir = Path(__file__).resolve().parent
    log_path = setup_logging(script_dir)

    OpenAI = import_openai_or_explain()
    api_key = resolve_api_key(script_dir)

    if str(INPUT_DIR) == "/path/to/your/images":
        print("\nERROR: You still need to set INPUT_DIR in the CONFIG section.")
        print("Example:")
        print('  INPUT_DIR = Path("/Users/yourname/Documents/my_images")\n')
        sys.exit(1)

    if str(OUTPUT_DIR) == "/path/to/output_json":
        print("\nERROR: You still need to set OUTPUT_DIR in the CONFIG section.")
        print("Example:")
        print('  OUTPUT_DIR = Path("/Users/yourname/Documents/output_json")\n')
        sys.exit(1)

    if not INPUT_DIR.exists():
        print(f"\nERROR: Input folder does not exist: {INPUT_DIR}\n")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key)

    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    image_files = sorted([p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in image_extensions])

    if not image_files:
        print(f"No supported image files found in: {INPUT_DIR}")
        print("Supported extensions: .png, .jpg, .jpeg, .tif, .tiff, .webp")
        return

    print(f"Found {len(image_files)} image(s).")
    print(f"Saving JSON files to: {OUTPUT_DIR}")
    print(f"Error log: {log_path}\n")

    processed = 0
    failed = 0

    for i, image_path in enumerate(image_files, start=1):
        print(f"[{i}/{len(image_files)}] Processing: {image_path.name}")

        try:
            data_url = image_to_data_url(image_path)

            response = client.responses.create(
                model=MODEL,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": PROMPT},
                            {"type": "input_image", "image_url": data_url}
                        ]
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "historical_document_reader",
                        "schema": SCHEMA,
                        "strict": True
                    }
                }
            )

            result_text = get_response_text(response)
            result_json = json.loads(result_text)

            out_path = OUTPUT_DIR / f"{image_path.stem}.json"
            out_path.write_text(
                json.dumps(result_json, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )

            processed += 1

        except Exception as e:
            failed += 1
            error_message = f"Failed on {image_path.name}: {e}"
            print(f"  ERROR: {error_message}")
            logging.exception(error_message)

    print("\nFinished.")
    print(f"Successful: {processed}")
    print(f"Failed: {failed}")

    if failed:
        print(f"See log file for details: {log_path}")


if __name__ == "__main__":
    main()