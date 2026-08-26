"""
MedVerax AI - Rule-Based Pattern Matcher
Detects suspicious medical phrases, absolute cure claims, and dangerous health advice.
"""
import re
from typing import List, Dict, Any

SUSPICIOUS_RULES = [
    # Absolute / Guaranteed Cure Claims
    {
        "pattern": r"\b(100% cure|100% effective|guaranteed cure|guarantees a complete cure|cure guaranteed|miracle cure|permanent cure|cures all|cure for cancer|cure for diabetes|instant cure|instantly cure|cure overnight|dissolve(s)? overnight|cures (stage \d|terminal|chronic))\b",
        "category": "Absolute Cure Claim",
        "severity": "high",
        "explanation": "Absolute cure claims are a major indicator of medical misinformation. Legitimate medical treatments rarely promise 100% guarantees or instantaneous cures."
    },
    # Medical Avoidance / Dangerous Advice
    {
        "pattern": r"\b(stop taking (your )?(medicine|pills|insulin|medication)|throw away your (inhaler|prescription)|no doctor needed|no doctor required|never visit a hospital|stop dialysis|avoid (doctors|hospitals|vaccines))\b",
        "category": "Dangerous Medical Avoidance",
        "severity": "high",
        "explanation": "Advising patients to discontinue prescribed medications or avoid medical professionals poses severe health risks and should never be followed without physician consultation."
    },
    # Universal / Panacea Claims
    {
        "pattern": r"\b(works for everyone|cures 102|cures all (diseases|illnesses)|panacea|universal cure|realigns? cellular dna|purifies (your )?blood 24/7)\b",
        "category": "Universal Panacea Claim",
        "severity": "medium",
        "explanation": "Human biology is complex; no single treatment, remedy, or diet cures all diseases or works identically for every individual."
    },
    # Conspiracy / Hidden Secret Tropes
    {
        "pattern": r"\b(doctors don'?t want you to know|big pharma is hiding|secret (remedy|cure|miracle|plant|root|frequency)|government[- ]banned supplement)\b",
        "category": "Medical Conspiracy Trope",
        "severity": "medium",
        "explanation": "Language suggesting secret or suppressed cures is a common rhetorical tactic in commercial health scams."
    },
    # Unverified / Pseudoscience Substances
    {
        "pattern": r"\b(colloidal silver|miracle mineral solution|hydrogen peroxide cures|baking soda (baths|cures)|raw apricot seeds|scalar pendant|quantum vibration water)\b",
        "category": "Unverified Alternative Remedy",
        "severity": "medium",
        "explanation": "This substance or device is frequently promoted with unsubstantiated medical efficacy and may carry unverified toxicological risks."
    }
]

def analyze_rules(cleaned_text: str) -> List[Dict[str, Any]]:
    """
    Evaluates cleaned text against regex rules and returns triggered patterns.
    """
    detected = []
    for rule in SUSPICIOUS_RULES:
        matches = re.findall(rule["pattern"], cleaned_text, flags=re.IGNORECASE)
        if matches:
            # Flatten matched group if tuples exist
            matched_phrases = []
            for m in matches:
                if isinstance(m, tuple):
                    matched_phrases.append(m[0])
                else:
                    matched_phrases.append(m)
            
            # Deduplicate phrases
            unique_phrases = list(set([p for p in matched_phrases if p]))
            detected.append({
                "category": rule["category"],
                "severity": rule["severity"],
                "matched_phrases": unique_phrases,
                "explanation": rule["explanation"]
            })
    return detected
