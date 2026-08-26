"""
MedVerax AI - FastAPI Main Application
"""
import os
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from backend.services.preprocessor import clean_text
from backend.services.rules import analyze_rules
from backend.services.explainer import evaluate_claim, MEDICAL_DISCLAIMER
from backend.database.db import init_db, save_analysis, get_history, clear_history

# Global variables for model artifacts
vectorizer = None
model = None

def load_artifacts():
    """Loads database schema and trained model artifacts."""
    global vectorizer, model
    init_db()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vec_path = os.path.join(base_dir, "model", "tfidf_vectorizer.joblib")
    model_path = os.path.join(base_dir, "model", "logistic_regression_model.joblib")
    
    if os.path.exists(vec_path) and os.path.exists(model_path):
        vectorizer = joblib.load(vec_path)
        model = joblib.load(model_path)
        print("[+] Machine learning model and TF-IDF vectorizer loaded successfully.")
    else:
        print("[!] Warning: Model files not found. Run 'python notebooks/train_model.py' to generate them.")

# Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield

# Initial direct load for immediate execution / serverless warm-up
load_artifacts()

# Initialize FastAPI App
app = FastAPI(
    title="MedVerax AI - Health Misinformation Detection API",
    description="Analyzes medical and health claims using TF-IDF, Logistic Regression, and rule-based explainability.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClaimRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000, description="Health claim text to analyze")

class ClaimResponse(BaseModel):
    claim: str
    risk_level: str
    prediction: str
    confidence: float
    detected_patterns: List[Dict[str, Any]]
    explanation: str
    safety_recommendation: str
    disclaimer: str

@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint to verify backend status."""
    return {
        "status": "healthy",
        "service": "MedVerax AI",
        "model_loaded": model is not None and vectorizer is not None,
        "version": "1.0.0"
    }

@app.post("/analyze", response_model=ClaimResponse, tags=["Analysis"])
def analyze_health_claim(payload: ClaimRequest):
    """
    Analyzes a health claim:
    1. Preprocesses and cleans input text
    2. Runs rule-based pattern matching
    3. Runs TF-IDF + Logistic Regression classification
    4. Aggregates risk score and generates explainability
    5. Saves the record to SQLite
    """
    global vectorizer, model
    
    raw_text = payload.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    cleaned = clean_text(raw_text)
    
    # 1. Rule-Based Analysis
    detected_rules = analyze_rules(cleaned)
    
    # 2. Machine Learning Inference
    ml_misinfo_prob = 0.50
    if vectorizer is not None and model is not None:
        try:
            vec_features = vectorizer.transform([cleaned])
            probabilities = model.predict_proba(vec_features)[0]
            ml_misinfo_prob = float(probabilities[1])
        except Exception as e:
            print(f"[!] Inference error: {e}")
            ml_misinfo_prob = 0.50
    
    # 3. Explainability & Risk Calculation
    result = evaluate_claim(
        raw_text=raw_text,
        cleaned_text=cleaned,
        ml_misinfo_prob=ml_misinfo_prob,
        detected_rules=detected_rules
    )
    
    # 4. Save to Database
    try:
        save_analysis(result)
    except Exception as e:
        print(f"[!] Database save error: {e}")
        
    return result

@app.get("/history", tags=["History"])
def fetch_history(limit: int = 20):
    """Retrieves previous claim analyses from SQLite."""
    return {"history": get_history(limit=limit)}

@app.delete("/history", tags=["History"])
def reset_history():
    """Clears previous claim analyses history."""
    clear_history()
    return {"message": "Analysis history cleared successfully."}

# Locate and serve frontend
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

@app.get("/", include_in_schema=False)
def serve_index():
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "MedVerax AI API is running. Visit /docs for API documentation."}

if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
