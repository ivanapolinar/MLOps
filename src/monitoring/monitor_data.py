import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

def generate_data_drift_report(
    reference_path="data/processed/steel_energy_clean.csv",
    output_path="reports/monitoring/data_drift_report.html",
):
    print("Loading dataset...")

    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference dataset not found: {reference_path}")

    df = pd.read_csv(reference_path)

    print("Generating Evidently Data Drift Report...")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df, current_data=df)

    report.save_html(output_path)

    print(f"Report saved at: {output_path}")

if __name__ == "__main__":
    generate_data_drift_report()
