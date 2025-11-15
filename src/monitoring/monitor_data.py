import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def load_data(path):
    """Load CSV file."""
    return pd.read_csv(path)


def generate_report(reference_path, current_path, output_path):
    """Generate drift report with Evidently."""
    reference = load_data(reference_path)
    current = load_data(current_path)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    report.save_html(output_path)
    print(f"Report saved to: {output_path}")
