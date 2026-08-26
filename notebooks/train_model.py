"""
MedVerax AI - Model Training Script
Trains a TF-IDF + Logistic Regression pipeline for health misinformation detection.
"""

import os
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def clean_text(text: str) -> str:
    """Preprocess and clean raw input text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s%]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("=" * 60)
    print(" MedVerax AI - Machine Learning Pipeline Training ")
    print("=" * 60)

    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_path = os.path.join(project_root, "data", "health_claims.csv")
    model_dir = os.path.join(project_root, "backend", "model")
    os.makedirs(model_dir, exist_ok=True)

    print(f"[*] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[*] Dataset shape: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"[*] Class distribution:\n{df['label'].value_counts().rename({0: 'Reliable (0)', 1: 'Potential Misinformation (1)'})}")

    # 1. Clean data
    print("\n[*] Preprocessing and cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    X = df['cleaned_text']
    y = df['label']

    # 2. Train / Test Split
    print("[*] Splitting dataset (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. TF-IDF Vectorization
    print("[*] Fitting TF-IDF Vectorizer (Unigrams + Bigrams)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=2500,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 4. Logistic Regression Model
    print("[*] Training Logistic Regression classifier...")
    model = LogisticRegression(
        C=2.0,
        solver='liblinear',
        random_state=42,
        max_iter=1000
    )
    model.fit(X_train_tfidf, y_train)

    # 5. Model Evaluation
    print("\n" + "=" * 60)
    print(" MODEL EVALUATION METRICS ")
    print("=" * 60)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy:        {acc * 100:.2f}%")
    print(f"Precision:       {prec * 100:.2f}%")
    print(f"Recall:          {rec * 100:.2f}%")
    print(f"F1-Score:        {f1 * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print("                   Pred Reliable (0)  Pred Misinfo (1)")
    print(f"Actual Reliable (0):       {cm[0][0]:<15} {cm[0][1]:<15}")
    print(f"Actual Misinfo (1):        {cm[1][0]:<15} {cm[1][1]:<15}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Reliable", "Potential Misinformation"]))

    # 6. Save Model Artifacts
    vec_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    model_path = os.path.join(model_dir, "logistic_regression_model.joblib")
    
    print(f"[*] Saving TF-IDF Vectorizer to: {vec_path}")
    joblib.dump(vectorizer, vec_path)

    print(f"[*] Saving Logistic Regression model to: {model_path}")
    joblib.dump(model, model_path)

    print("\n[+] Training complete! Model artifacts successfully generated.\n")

if __name__ == "__main__":
    main()
