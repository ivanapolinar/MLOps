import requests
import json

BASE_URL = "http://localhost:8000"

def pretty_print(response):
    try:
        print(json.dumps(response.json(), indent=4, ensure_ascii=False))
    except Exception as e:
        print("Error al decodificar respuesta:", e, response.text)

def test_health():
    print("Probando /health...")
    r = requests.get(f"{BASE_URL}/health")
    print("Status:", r.status_code)
    pretty_print(r)

def test_version():
    print("\nProbando /version...")
    r = requests.get(f"{BASE_URL}/version")
    print("Status:", r.status_code)
    pretty_print(r)

def test_predict():
    print("\nProbando /predict (predicción individual)...")
    data = {
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
    r = requests.post(f"{BASE_URL}/predict", json=data)
    print("Status:", r.status_code)
    pretty_print(r)

def test_batch_predict():
    print("\nProbando /batch_predict (predicción por lote)...")
    data = {
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
    r = requests.post(f"{BASE_URL}/batch_predict", json=data)
    print("Status:", r.status_code)
    pretty_print(r)

def test_metrics():
    print("\nProbando /metrics...")
    r = requests.get(f"{BASE_URL}/metrics")
    print("Status:", r.status_code)
    pretty_print(r)

def test_retrain():
    print("\nProbando /retrain (dummy)...")
    r = requests.post(f"{BASE_URL}/retrain")
    print("Status:", r.status_code)
    pretty_print(r)

if __name__ == "__main__":
    print("Iniciando pruebas de la API Steel Energy...\n")
    test_health()
    test_version()
    test_predict()
    test_batch_predict()
    test_metrics()
    test_retrain()