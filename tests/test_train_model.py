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
    # Ensure at least 3 observations for class
    data = {
        'f1': [1, 2, 3, 4, 5, 6, 7, 8],
        'f2': [2, 3, 3, 4, 2, 1, 2, 4],
        'cat': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'Load_Type': [0, 1, 1, 0, 1, 0, 1, 0],
        'date': pd.date_range('2022-01-01', periods=8, freq='D')
    }
    return pd.DataFrame(data)


def test_load_data(tmp_path, dummy_df):
    """
    Prueba: load_data carga el CSV correctamente.
    """
    csv_path = tmp_path / "test.csv"
    dummy_df.to_csv(csv_path, index=False)
    df_loaded = train_model.load_data(str(csv_path))
    # Convert 'date' to datetime in both to compare
    df_loaded['date'] = pd.to_datetime(df_loaded['date'])
    dummy_df['date'] = pd.to_datetime(dummy_df['date'])
    pd.testing.assert_frame_equal(df_loaded, dummy_df)


def test_split_data(dummy_df):
    """
    Prueba: split_data divide correctamente el DataFrame.
    """
    X_train, X_test, y_train, y_test = train_model.split_data(
        dummy_df,
        target="Load_Type",
        test_size=0.33,
        random_state=0
    )
    # Check if sum corresponds
    assert len(X_train) + len(X_test) == len(dummy_df)
    # Check if target was correctly split
    assert all(~X_train.columns.isin(['Load_Type', 'date']))
    assert set(y_train.unique()).issubset({0, 1})


def test_build_preprocessing(dummy_df):
    """
    Prueba: build_preprocessing identifica correctamente columnas.
    """
    X = dummy_df.drop(columns=['Load_Type', 'date'])
    preprocessing, num_cols, cat_cols = train_model.build_preprocessing(X)
    assert 'f1' in num_cols and 'f2' in num_cols
    assert 'cat' in cat_cols
    # Check that it is a ColumnTransformer
    from sklearn.compose import ColumnTransformer
    assert isinstance(preprocessing, ColumnTransformer)


def test_train_base_model(dummy_df):
    """
    Prueba: train_base_model entrena y ajusta el modelo.
    """
    X_train, X_test, y_train, y_test = train_model.split_data(
        dummy_df,
        target="Load_Type"
    )
    preprocessing, num_cols, cat_cols = train_model.build_preprocessing(
        X_train
    )
    model = train_model.train_base_model(
        X_train,
        y_train,
        preprocessing
    )
    # Check if model is trained
    check_is_fitted(model.named_steps['randomforestclassifier'])
    # Predict without errors
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test)


def test_evaluate_model(tmp_path, dummy_df):
    """
    Prueba: evaluate_model calcula métricas y guarda la figura.
    """
    X_train, X_test, y_train, y_test = train_model.split_data(
        dummy_df,
        target="Load_Type"
    )
    preprocessing, num_cols, cat_cols = train_model.build_preprocessing(
        X_train
    )
    model = train_model.train_base_model(
        X_train,
        y_train,
        preprocessing
    )
    figures_dir = tmp_path
    acc, fig_path, report = train_model.evaluate_model(
        model,
        X_test,
        y_test,
        str(figures_dir),
        name="unit"
    )
    assert 0.0 <= acc <= 1.0
    assert os.path.exists(fig_path)
    assert isinstance(report, dict)


def test_hyperparameter_tuning(dummy_df):
    """
    Prueba: hyperparameter_tuning devuelve un estimador entrenado y el dict de hiperparámetros.
    """
    X_train, X_test, y_train, y_test = train_model.split_data(
        dummy_df,
        target="Load_Type"
    )
    preprocessing, num_cols, cat_cols = train_model.build_preprocessing(
        X_train
    )
    base_model = train_model.train_base_model(
        X_train,
        y_train,
        preprocessing
    )
    best_model, best_params = train_model.hyperparameter_tuning(
        base_model,
        X_train,
        y_train
    )
    # Check that the model is fitted and returns parameters
    check_is_fitted(best_model.named_steps['randomforestclassifier'])
    assert isinstance(best_params, dict)
    assert "randomforestclassifier__n_estimators" in best_params


def test_save_feature_importance(tmp_path, dummy_df):
    """
    Prueba: save_feature_importance guarda archivos de importancia de variables.
    """
    X_train, X_test, y_train, y_test = train_model.split_data(
        dummy_df,
        target="Load_Type"
    )
    preprocessing, num_cols, cat_cols = train_model.build_preprocessing(
        X_train
    )
    model = train_model.train_base_model(
        X_train,
        y_train,
        preprocessing
    )
    fi_path, top_feat_path = train_model.save_feature_importance(
        model,
        num_cols,
        cat_cols,
        str(tmp_path)
    )
    assert os.path.exists(fi_path)
    assert os.path.exists(top_feat_path)
    df_fi = pd.read_csv(fi_path)
    assert "Feature" in df_fi.columns
    assert "Importance" in df_fi.columns


def test_save_model(tmp_path, dummy_df):
    """
    Prueba: save_model guarda el modelo correctamente.
    """
    X_train, X_test, y_train, y_test = train_model.split_data(
        dummy_df,
        target="Load_Type"
    )
    preprocessing, num_cols, cat_cols = train_model.build_preprocessing(
        X_train
    )
    model = train_model.train_base_model(
        X_train,
        y_train,
        preprocessing
    )
    model_path = tmp_path / "model.joblib"
    train_model.save_model(model, str(model_path))
    assert os.path.exists(model_path)
