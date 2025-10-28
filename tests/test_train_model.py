import os
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted
from src.models import train_model


@pytest.fixture
def dummy_df():
    """
    Fixture: crea un DataFrame de prueba para todas las funciones.
    """
    data = {
        'f1': [1, 2, 3, 4, 5, 6, 7, 8],
        'f2': [2, 3, 3, 4, 2, 1, 2, 4],
        'cat': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'Load_Type': [0, 1, 1, 0, 1, 0, 1, 0],
        'date': pd.date_range('2022-01-01', periods=8, freq='D')
    }
    return pd.DataFrame(data)


def test_data_repository_load(tmp_path, dummy_df):
    """
    Prueba: load_data carga el CSV correctamente.
    """
    csv_path = tmp_path / "test.csv"
    dummy_df.to_csv(csv_path, index=False)
    repo = train_model.DataRepository()
    df_loaded = repo.load(str(csv_path))
    df_loaded['date'] = pd.to_datetime(df_loaded['date'])
    dummy_df['date'] = pd.to_datetime(dummy_df['date'])
    pd.testing.assert_frame_equal(df_loaded, dummy_df)


def test_data_splitter(dummy_df):
    """
    Prueba: split_data divide correctamente el DataFrame.
    """
    splitter = train_model.DataSplitter()
    X_train, X_test, y_train, y_test = splitter.split(dummy_df)
    assert len(X_train) + len(X_test) == len(dummy_df)
    assert all(~X_train.columns.isin(['Load_Type', 'date']))
    assert set(y_train.unique()).issubset({0, 1})


def test_preprocessing_builder(dummy_df):
    """
    Prueba: build_preprocessing identifica correctamente columnas.
    """
    builder = train_model.PreprocessingBuilder()
    X = dummy_df.drop(columns=['Load_Type', 'date'])
    preprocessing, cat_cols, num_cols = builder.build(X)
    assert 'f1' in num_cols and 'f2' in num_cols
    assert 'cat' in cat_cols


def test_model_trainer(dummy_df):
    """
    Prueba: train_base_model entrena y ajusta el modelo.
    """
    splitter = train_model.DataSplitter()
    builder = train_model.PreprocessingBuilder()
    trainer = train_model.ModelTrainer()
    X_train, X_test, y_train, y_test = splitter.split(dummy_df)
    preprocessing, cat_cols, num_cols = builder.build(X_train)
    model = trainer.fit(X_train, y_train, preprocessing)
    check_is_fitted(model.named_steps['randomforestclassifier'])
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test)


def test_model_evaluator(tmp_path, dummy_df):
    """
    Prueba: evaluate_model calcula métricas y guarda la figura.
    """
    splitter = train_model.DataSplitter()
    builder = train_model.PreprocessingBuilder()
    trainer = train_model.ModelTrainer()
    evaluator = train_model.ModelEvaluator()
    X_train, X_test, y_train, y_test = splitter.split(dummy_df)
    preprocessing, cat_cols, num_cols = builder.build(X_train)
    model = trainer.fit(X_train, y_train, preprocessing)
    figures_dir = tmp_path
    acc, fig_path, report = evaluator.evaluate(
        model,
        X_test,
        y_test,
        str(figures_dir),
        name="unit"
    )
    assert 0.0 <= acc <= 1.0
    assert os.path.exists(fig_path)
    assert isinstance(report, str)


def test_hyperparameter_tuner(dummy_df):
    """
    Prueba: hyperparameter_tuning devuelve estimador entrenado y parámetros.
    """
    splitter = train_model.DataSplitter()
    builder = train_model.PreprocessingBuilder()
    trainer = train_model.ModelTrainer()
    tuner = train_model.HyperparameterTuner()
    X_train, X_test, y_train, y_test = splitter.split(dummy_df)
    preprocessing, cat_cols, num_cols = builder.build(X_train)
    base_model = trainer.fit(X_train, y_train, preprocessing)
    best_model, best_params = tuner.tune(
        base_model,
        X_train,
        y_train,
        random_state=42
    )
    check_is_fitted(best_model.named_steps['randomforestclassifier'])
    assert isinstance(best_params, dict)
    assert "randomforestclassifier__n_estimators" in best_params


def test_feature_importance_exporter(tmp_path, dummy_df):
    """
    Prueba: save_feature_importance guarda archivos de importancia.
    """
    splitter = train_model.DataSplitter()
    builder = train_model.PreprocessingBuilder()
    trainer = train_model.ModelTrainer()
    exporter = train_model.FeatureImportanceExporter()
    X_train, X_test, y_train, y_test = splitter.split(dummy_df)
    preprocessing, cat_cols, num_cols = builder.build(X_train)
    model = trainer.fit(X_train, y_train, preprocessing)
    fi_path, top_feat_path = exporter.export(model, str(tmp_path))
    assert os.path.exists(fi_path)
    assert os.path.exists(top_feat_path)
    df_fi = pd.read_csv(fi_path)
    assert "Feature" in df_fi.columns
    assert "Importance" in df_fi.columns


def test_model_persister(tmp_path, dummy_df):
    """
    Prueba: save_model guarda el modelo correctamente.
    """
    splitter = train_model.DataSplitter()
    builder = train_model.PreprocessingBuilder()
    trainer = train_model.ModelTrainer()
    persister = train_model.ModelPersister()
    X_train, X_test, y_train, y_test = splitter.split(dummy_df)
    preprocessing, cat_cols, num_cols = builder.build(X_train)
    model = trainer.fit(X_train, y_train, preprocessing)
    model_path = tmp_path / "model.joblib"
    persister.save(model, str(model_path))
    assert os.path.exists(model_path)
