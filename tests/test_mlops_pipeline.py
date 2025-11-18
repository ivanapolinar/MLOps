import os
import pandas as pd
from src.pipeline.mlops_pipeline import MLOpsPipeline


def _make_raw_df_for_cleaning():
    # Fechas en formato d/m/Y H:M para cumplir con el parser
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


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    # Configurar MLflow en file store temporal
    mlruns_dir = tmp_path / 'mlruns'
    pipe = MLOpsPipeline(
        mlflow_tracking_uri=f'file:{mlruns_dir}',
        mlflow_experiment='steel_energy_test',
        register_in_registry=False,
    )

    # 1) Procesamiento de datos crudos → limpios
    raw_csv = tmp_path / 'raw.csv'
    clean_csv = tmp_path / 'clean.csv'
    _make_raw_df_for_cleaning().to_csv(raw_csv, index=False)
    df_clean = pipe.process_raw_to_clean(str(raw_csv), str(clean_csv))
    assert os.path.exists(clean_csv)
    assert not df_clean['Load_Type'].isna().any()

    # 2) Entrenamiento y exportación de artefactos
    model_path = tmp_path / 'models' / 'final_model.joblib'
    figures_dir = tmp_path / 'figures'
    acc_opt, fi_csv, top_png = pipe.run_training_pipeline(
        str(clean_csv), str(model_path), str(figures_dir)
    )
    assert os.path.exists(model_path)
    assert os.path.exists(fi_csv)
    assert os.path.exists(top_png)
    assert 0.0 <= acc_opt <= 1.0

    # 3) Predicción por lote y métricas/figura
    predictions_path = tmp_path / 'preds.csv'
    metrics_path = tmp_path / 'metrics.json'
    pipe.batch_predict(
        str(clean_csv),
        str(model_path),
        str(predictions_path),
        str(metrics_path),
        str(figures_dir),
    )
    assert os.path.exists(predictions_path)
    assert os.path.exists(metrics_path)
    # Figura de confusión generada durante la predicción
    cm_path = os.path.join(str(figures_dir), 'confusion_matrix_predict.png')
    assert os.path.exists(cm_path)
