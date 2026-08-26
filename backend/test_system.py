"""
MedVerax AI - Verification Test Script
"""
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_system():
    print("[*] Running Verification Tests on MedVerax AI...\n")
    
    # 1. Test Health Endpoint
    res_health = client.get("/health")
    print(f"1. Health Check: Status {res_health.status_code} -> {res_health.json()}")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"
    assert res_health.json()["model_loaded"] is True
    
    # 2. Test High-Risk Misinformation Claim
    sample_misinfo = "This herbal tea guarantees a 100% cure for cancer in 2 weeks with no doctor needed."
    res_misinfo = client.post("/analyze", json={"text": sample_misinfo})
    print(f"\n2. Misinformation Test: Status {res_misinfo.status_code}")
    data_m = res_misinfo.json()
    print(f"   Risk Level:    {data_m['risk_level']}")
    print(f"   Confidence:    {data_m['confidence']}")
    print(f"   Patterns:      {[p['category'] for p in data_m['detected_patterns']]}")
    print(f"   Explanation:   {data_m['explanation']}")
    assert data_m["risk_level"] == "High Risk"
    assert len(data_m["detected_patterns"]) >= 1
    
    # 3. Test Reliable Medical Statement
    sample_reliable = "Regular moderate aerobic exercise and balanced dietary habits improve cardiovascular health."
    res_reliable = client.post("/analyze", json={"text": sample_reliable})
    print(f"\n3. Reliable Test: Status {res_reliable.status_code}")
    data_r = res_reliable.json()
    print(f"   Risk Level:    {data_r['risk_level']}")
    print(f"   Confidence:    {data_r['confidence']}")
    print(f"   Prediction:    {data_r['prediction']}")
    assert data_r["risk_level"] == "Low Risk"
    
    # 4. Test History Endpoint
    res_history = client.get("/history")
    print(f"\n4. History Check: Status {res_history.status_code}")
    history_records = res_history.json()["history"]
    print(f"   Total stored records: {len(history_records)}")
    assert len(history_records) >= 2
    
    print("\n[+] ALL SYSTEM TESTS PASSED SUCCESSFULLY! The application is fully functional.\n")

if __name__ == "__main__":
    test_system()
