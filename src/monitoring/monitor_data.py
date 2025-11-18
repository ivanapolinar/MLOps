import pandas as pd
import os


def load_monitoring_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Monitoring file not found: {path}"
        )
    return pd.read_csv(path)


def generate_summary(df):
    summary = df.describe(include="all")
    return summary


def save_summary(summary, output_path):
    summary.to_csv(output_path)
    print(f"Summary saved to {output_path}")
