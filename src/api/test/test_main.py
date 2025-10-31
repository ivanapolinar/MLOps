from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

# Example valid payload for prediction
valid_payload = {
    "Usage_kWh": 200.0,
    "Lagging_Current_Reactive.Power_kVarh": 12.1,
    "Leading_Current_Reactive_Power_kVarh": 0.0,
    "CO2(tCO2)": 0.07,
    "Lagging_Current_Power_Factor": 0.85,
    "Leading_Current_Power_Factor": 0.0,
    "NSM": 36000,
    "mixed_type_col": 0.0,
    "WeekStatus": "WEEKDAY",
    "Day_of_week": "MONDAY"
}

batch_payload = {
    "records": [
        {
            "Usage_kWh": 200.0,
            "Lagging_Current_Reactive.Power_kVarh": 12.1,
            "Leading_Current_Reactive_Power_kVarh": 0.0,
            "CO2(tCO2)": 0.07,
            "Lagging_Current_Power_Factor": 0.85,
            "Leading_Current_Power_Factor": 0.0,
            "NSM": 36000,
            "mixed_type_col": 0.0,
            "WeekStatus": "WEEKDAY",
            "Day_of_week": "MONDAY"
        },
        {
            "Usage_kWh": 250.0,
            "Lagging_Current_Reactive.Power_kVarh": 15.0,
            "Leading_Current_Reactive_Power_kVarh": 5.0,
            "CO2(tCO2)": 0.09,
            "Lagging_Current_Power_Factor": 0.80,
            "Leading_Current_Power_Factor": 0.10,
            "NSM": 40000,
            "mixed_type_col": 1.0,
            "WeekStatus": "WEEKEND",
            "Day_of_week": "SATURDAY"
        }
    ]
}

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()
    assert "model_loaded" in resp.json()

def test_version():
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert "version" in body
    assert "model_path" in body

def test_predict_valid():
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert "probabilities" in body

def test_batch_predict_valid():
    resp = client.post("/batch_predict", json=batch_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert all("prediction" in r for r in body)
    assert all("probabilities" in r for r in body)

def test_metrics():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "metrics" in body

def test_retrain():
    resp = client.post("/retrain")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body