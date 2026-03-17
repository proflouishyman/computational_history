# Shallow Reading Workflow

## Purpose

Convert archival document images into one conservative, structured JSON file per image.

## Script

- `code/shallow_reading_local.py`

## Input/Output contract

- Input directory: `./images` (relative to where the script runs)
- Output directory: `./output_json` (one `.json` per image stem)
- Supported image types: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.webp`

## JSON schema shape

- `file_summary`: string
- `transcription`: string
- `named_entities`: object with lists for `people`, `places`, `organizations`, `dates`, `addresses`, `money`
- `categories`: booleans for domain categories plus `other` list
- `key_facts`: list of strings
- `language`: string
- `document_type`: string
- `confidence_notes`: string

## Notes

- Extraction is intentionally literal and conservative.
- Error details are logged to `image_to_json_errors.log`.
