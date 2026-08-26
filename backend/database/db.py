"""
MedVerax AI - SQLite Database Service
Manages persistence of user queries, analysis outcomes, and timestamps.
"""
import os
import json
import sqlite3
from typing import List, Dict, Any

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "analyses.db")

def get_connection():
    """Returns a SQLite connection with dict-like row access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema if not already present."""
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

def save_analysis(data: Dict[str, Any]) -> int:
    """Inserts a completed claim analysis into SQLite."""
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

def get_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Fetches the latest analysis history records."""
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

def clear_history():
    """Clears all records from the analyses table."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analyses")
        conn.commit()
