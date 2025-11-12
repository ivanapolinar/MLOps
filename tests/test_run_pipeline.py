import os
import pandas as pd

from click.testing import CliRunner

from src.pipeline.run_pipeline import main as run_cli


def _make_raw_df():
    dates = [
        "01/01/2022 00:00",
        "01/01/2022 00:15",
        "01/01/2022 00:30",
        "01/01/2022 00:45",
        "01/01/2022 01:00",
        "01/01/2022 01:15",
        "01/01/2022 01:30",
        "01/01/2022 01:45",
    ]
    return pd.DataFrame({
        'date': dates,
        'Usage_kWh': [10, 12, 11, 13, 9, 8, 14, 15],
        'f1': [1, 2, 3, 4, 5, 6, 7, 8],
        'f2': [2, 3, 3, 4, 2, 1, 2, 4],
        'cat': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'WeekStatus': ['WEEKDAY'] * 8,
        'Day_of_week': ['SATURDAY'] * 8,
        'Load_Type': [0, 1, 1, 0, 1, 0, 1, 0],
    })


def test_run_pipeline_cli_end_to_end(tmp_path, monkeypatch):
    # Backend gráfico sin interfaz y MLflow local por archivo
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    mlruns_dir = tmp_path / 'mlruns'
    monkeypatch.setenv('MLFLOW_TRACKING_URI', f'file:{mlruns_dir}')
    monkeypatch.setenv('MLFLOW_EXPERIMENT', 'steel_energy_cli_test')

    # Archivos/dirs de trabajo
    raw_csv = tmp_path / 'raw.csv'
    clean_csv = tmp_path / 'clean.csv'
    model_path = tmp_path / 'model.joblib'
    predictions = tmp_path / 'preds.csv'
    metrics = tmp_path / 'metrics.json'
    figures = tmp_path / 'figures'

    _make_raw_df().to_csv(raw_csv, index=False)

    # Atajo para acelerar: monkeypatch de tuning
    # Sustituimos el método del pipeline para devolver el base_model
    import src.pipeline.mlops_pipeline as pl

    def fast_tune(self, model, X_train, y_train):
        return model, {'mock_tuning': True}

    monkeypatch.setattr(pl.MLOpsPipeline, 'tune_hyperparams', fast_tune)

    runner = CliRunner()
    result = runner.invoke(
        run_cli,
        [
            '--raw-input', str(raw_csv),
            '--clean-output', str(clean_csv),
            '--model-path', str(model_path),
            '--predictions', str(predictions),
            '--metrics', str(metrics),
            '--figures', str(figures),
            '--mlflow-uri', f'file:{mlruns_dir}',
            '--mlflow-experiment', 'steel_energy_cli_test',
            '--no-register',
        ],
    )

    assert result.exit_code == 0, result.output
    # Salidas esperadas
    assert os.path.exists(clean_csv)
    assert os.path.exists(model_path)
    assert os.path.exists(predictions)
    assert os.path.exists(metrics)
    assert os.path.exists(os.path.join(figures, 'confusion_matrix_base.png'))
    assert os.path.exists(os.path.join(figures, 'confusion_matrix_optimized.png'))
    assert os.path.exists(os.path.join(figures, 'top_features.png'))
    assert os.path.exists(
        os.path.join(figures, 'confusion_matrix_predict.png')
    )

