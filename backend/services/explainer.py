"""
MedVerax AI - Explainability & Risk Aggregator Service
"""
from typing import List, Dict, Any

MEDICAL_DISCLAIMER = (
    "This system is for information and awareness only. "
    "It does not provide medical diagnosis or replace a qualified healthcare professional."
)

def evaluate_claim(
    raw_text: str,
    cleaned_text: str,
    ml_misinfo_prob: float,
    detected_rules: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Combines Machine Learning prediction probabilities with rule-based pattern matching
    to determine risk level, explainability insights, and safety recommendations.
    """
    has_high_severity_rule = any(r["severity"] == "high" for r in detected_rules)
    has_medium_severity_rule = any(r["severity"] == "medium" for r in detected_rules)
    rule_count = len(detected_rules)

    # Risk Level Determination
    if has_high_severity_rule or ml_misinfo_prob >= 0.70:
        risk_level = "High Risk"
        prediction = "Potential misinformation"
        confidence = max(ml_misinfo_prob, 0.88 if has_high_severity_rule else ml_misinfo_prob)
    elif has_medium_severity_rule or ml_misinfo_prob >= 0.40:
        risk_level = "Medium Risk"
        prediction = "Needs verification / Potential exaggeration"
        confidence = max(ml_misinfo_prob, 0.65)
    else:
        risk_level = "Low Risk"
        prediction = "Consistent with reliable medical patterns"
        confidence = 1.0 - ml_misinfo_prob

    # Generate Explainability Narrative
    reasons = []
    if detected_rules:
        for r in detected_rules:
            phrases_str = ", ".join([f'"{p}"' for p in r["matched_phrases"]])
            reasons.append(f"Contains {r['category'].lower()} phrases ({phrases_str}). {r['explanation']}")
    
    if ml_misinfo_prob >= 0.65 and not detected_rules:
        reasons.append(
            "The statistical machine learning model identified linguistic markers and wording styles frequently present in unverified or sensationalized online health claims."
        )
    elif ml_misinfo_prob < 0.35 and not detected_rules:
        reasons.append(
            "The phrasing aligns with standard clinical or evidence-based descriptions commonly found in verified health literature."
        )

    if not reasons:
        reasons.append("The statement presents health information that warrants cautious independent verification with peer-reviewed medical guidance.")

    explanation = " ".join(reasons)

    # Safety Recommendations
    if risk_level == "High Risk":
        recommendation = (
            "Caution: This claim exhibits high-risk indicators (such as absolute guarantees or encouragement to bypass medical care). "
            "Never discontinue prescribed therapies or start alternative treatments without direct guidance from a licensed physician."
        )
    elif risk_level == "Medium Risk":
        recommendation = (
            "Verification Advised: This statement may contain exaggerated benefits or lack conclusive clinical evidence. "
            "Cross-reference this claim with authoritative medical databases (e.g., WHO, CDC, PubMed, or NHS) before acting on it."
        )
    else:
        recommendation = (
            "Good Practice: The statement appears consistent with general evidence-based health concepts. "
            "Always tailor general health advice to your individual medical profile in consultation with your doctor."
        )

    return {
        "claim": raw_text,
        "cleaned_text": cleaned_text,
        "risk_level": risk_level,
        "prediction": prediction,
        "confidence": round(float(confidence), 2),
        "ml_misinfo_probability": round(float(ml_misinfo_prob), 2),
        "detected_patterns": detected_rules,
        "explanation": explanation,
        "safety_recommendation": recommendation,
        "disclaimer": MEDICAL_DISCLAIMER
    }
