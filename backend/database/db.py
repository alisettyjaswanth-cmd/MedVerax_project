"""
MedVerax AI - SQLite Database Service
Manages persistence of user queries, analysis outcomes, and timestamps.
Compatible with local environments and Serverless platforms (Vercel / AWS Lambda).
"""
import os
import json
import tempfile
import sqlite3
from typing import List, Dict, Any

# Determine appropriate DB path (Vercel/Serverless uses /tmp because root FS is read-only)
if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    DB_PATH = os.path.join(tempfile.gettempdir(), "analyses.db")
else:
    DB_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(DB_DIR, "analyses.db")

def get_connection():
    """Returns a SQLite connection with dict-like row access."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        print(f"[!] Warning: Could not connect to {DB_PATH}, falling back to in-memory: {e}")
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        return conn

def init_db():
    """Initializes the database schema if not already present."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    claim_text TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    detected_patterns TEXT,
                    explanation TEXT,
                    safety_recommendation TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    except Exception as e:
        print(f"[!] Database init notice: {e}")

def save_analysis(data: Dict[str, Any]) -> int:
    """Inserts a completed claim analysis into SQLite."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analyses (
                    claim_text,
                    risk_level,
                    prediction,
                    confidence,
                    detected_patterns,
                    explanation,
                    safety_recommendation
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                data.get("claim", ""),
                data.get("risk_level", "Unknown"),
                data.get("prediction", ""),
                float(data.get("confidence", 0.0)),
                json.dumps(data.get("detected_patterns", [])),
                data.get("explanation", ""),
                data.get("safety_recommendation", "")
            ))
            conn.commit()
            return cursor.lastrowid
    except Exception as e:
        print(f"[!] Database save notice: {e}")
        return -1

def get_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Fetches the latest analysis history records."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, claim_text, risk_level, prediction, confidence, 
                       detected_patterns, explanation, safety_recommendation, created_at
                FROM analyses
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            
            history = []
            for row in rows:
                try:
                    patterns = json.loads(row["detected_patterns"]) if row["detected_patterns"] else []
                except Exception:
                    patterns = []
                history.append({
                    "id": row["id"],
                    "claim": row["claim_text"],
                    "risk_level": row["risk_level"],
                    "prediction": row["prediction"],
                    "confidence": row["confidence"],
                    "detected_patterns": patterns,
                    "explanation": row["explanation"],
                    "safety_recommendation": row["safety_recommendation"],
                    "created_at": row["created_at"]
                })
            return history
    except Exception as e:
        print(f"[!] Database fetch notice: {e}")
        return []

def clear_history():
    """Clears all records from the analyses table."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM analyses")
            conn.commit()
    except Exception as e:
        print(f"[!] Database clear notice: {e}")
