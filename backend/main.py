"""
MedVerax AI - FastAPI Main Application
"""
import sys
import os

# Robust path handling for local and cloud serverless runtimes
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for p in [PROJECT_ROOT, CURRENT_DIR, os.getcwd()]:
    if p not in sys.path:
        sys.path.insert(0, p)

import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

def train_in_memory():
    """Trains a fallback TF-IDF + Logistic Regression model in ~20ms if serialized files differ across Python versions."""
    global vectorizer, model
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        import pandas as pd
        
        candidates = [
            os.path.join(PROJECT_ROOT, "data", "health_claims.csv"),
            os.path.join(CURRENT_DIR, "data", "health_claims.csv"),
            os.path.join(CURRENT_DIR, "..", "data", "health_claims.csv"),
            os.path.join(os.getcwd(), "data", "health_claims.csv")
        ]
        csv_file = next((c for c in candidates if os.path.exists(c)), None)
        if csv_file:
            df = pd.read_csv(csv_file)
            cleaned_texts = [clean_text(str(t)) for t in df['text']]
            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=2500, sublinear_tf=True)
            X = vec.fit_transform(cleaned_texts)
            clf = LogisticRegression(C=2.0, solver='liblinear', random_state=42)
            clf.fit(X, df['label'])
            vectorizer = vec
            model = clf
            print("[+] In-memory model trained successfully on serverless startup.")
    except Exception as e:
        print(f"[!] Fallback training notice: {e}")

def load_artifacts():
    """Loads database schema and trained model artifacts with fallback."""
    global vectorizer, model
    try:
        init_db()
    except Exception as e:
        print(f"[!] init_db notice: {e}")
        
    loaded = False
    for base in [CURRENT_DIR, PROJECT_ROOT, os.getcwd()]:
        vec_path = os.path.join(base, "backend", "model", "tfidf_vectorizer.joblib")
        if not os.path.exists(vec_path):
            vec_path = os.path.join(base, "model", "tfidf_vectorizer.joblib")
            
        model_path = os.path.join(base, "backend", "model", "logistic_regression_model.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(base, "model", "logistic_regression_model.joblib")
            
        if os.path.exists(vec_path) and os.path.exists(model_path):
            try:
                vectorizer = joblib.load(vec_path)
                model = joblib.load(model_path)
                loaded = True
                print("[+] Models loaded successfully from disk.")
                break
            except Exception as err:
                print(f"[!] joblib load error (likely Python/scikit-learn version difference): {err}")
    
    if not loaded or vectorizer is None or model is None:
        train_in_memory()

# Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield

# Warm up artifacts on import
load_artifacts()

# Initialize FastAPI App
app = FastAPI(
    title="MedVerax AI - Health Misinformation Detection API",
    description="Analyzes medical and health claims using TF-IDF, Logistic Regression, and rule-based explainability.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
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
    """Analyzes a health claim."""
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

# Locate frontend directory
def find_frontend_dir():
    candidates = [
        os.path.join(PROJECT_ROOT, "frontend"),
        os.path.join(CURRENT_DIR, "..", "frontend"),
        os.path.join(CURRENT_DIR, "frontend"),
        os.path.join(os.getcwd(), "frontend")
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "index.html")):
            return c
    return None

frontend_dir = find_frontend_dir()

@app.get("/", include_in_schema=False)
def serve_index():
    if frontend_dir:
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
    return {"message": "MedVerax AI API is running! Visit /docs for API documentation."}

@app.get("/style.css", include_in_schema=False)
def serve_css():
    if frontend_dir:
        f = os.path.join(frontend_dir, "style.css")
        if os.path.exists(f):
            return FileResponse(f, media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found")

@app.get("/script.js", include_in_schema=False)
def serve_js():
    if frontend_dir:
        f = os.path.join(frontend_dir, "script.js")
        if os.path.exists(f):
            return FileResponse(f, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="script.js not found")
