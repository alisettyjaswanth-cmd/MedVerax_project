"""
MedVerax AI - Text Preprocessor Service
"""
import re

def clean_text(text: str) -> str:
    """
    Cleans and normalizes incoming raw text for machine learning inference and rule matching.
    """
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove web URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Retain alphanumeric characters, percent signs, and basic spaces
    text = re.sub(r'[^a-zA-Z0-9\s%]', ' ', text)
    # Collapse multiple whitespaces into a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text
