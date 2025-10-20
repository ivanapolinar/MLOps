import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_data(input_path):
    df = pd.read_csv(input_path)
    return df

def split_data(df, target="Load_Type", test_size=0.2, random_state=42):
    y = df[target]
    X = df.drop(columns=[target, 'date'], errors="ignore")
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def build_preprocessing(X):
    cat_cols = list(X.select_dtypes('object').columns)
    num_cols = list(X.select_dtypes('number').columns)
    preprocessing = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    return preprocessing, num_cols, cat_cols

def train_base_model(X_train, y_train, preprocessing):
    rf_model = make_pipeline(preprocessing, RandomForestClassifier(random_state=42))
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test, figures_dir, name="base"):
    y_pred = model.predict(X_test)
    print(f"Reporte de clasificación ({name}):")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de confusión - RandomForest ({name})")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/confusion_matrix_{name}.png")
    plt.close()

def hyperparameter_tuning(model, X_train, y_train):
    param_dist = {
        "randomforestclassifier__n_estimators": [100, 200, 400],
        "randomforestclassifier__max_depth": [10, 20, None],
        "randomforestclassifier__min_samples_split": [2, 5, 10],
        "randomforestclassifier__min_samples_leaf": [1, 2, 4],
        "randomforestclassifier__max_features": ["sqrt", "log2"]
    }
    rf_random = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=30,
        cv=3,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    rf_random.fit(X_train, y_train)
    print("Mejores hiperparámetros encontrados:")
    print(rf_random.best_params_)
    return rf_random.best_estimator_

def save_feature_importance(model, num_cols, cat_cols, figures_dir):
    rf_final = model.named_steps["randomforestclassifier"]
    ohe = model.named_steps["columntransformer"].named_transformers_["cat"]
    encoded_cat_cols = ohe.get_feature_names_out(cat_cols)
    final_feature_names = np.concatenate([num_cols, encoded_cat_cols])
    importances = rf_final.feature_importances_
    feature_importances = pd.DataFrame({
        "Feature": final_feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    feature_importances.head(15).to_csv(f"{figures_dir}/feature_importances.csv", index=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importances.head(15), x="Importance", y="Feature")
    plt.title("Top 15 Variables más importantes - RandomForest")
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/top_features.png")
    plt.close()

def save_model(model, model_path):
    joblib.dump(model, model_path)
    print(f"Modelo guardado en {model_path}")

def main(input_path="data/clean/steel_energy_clean.csv", model_path="models/best_rf_model.joblib", figures_dir="../../reports/figures"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    df = load_data(input_path)
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessing, num_cols, cat_cols = build_preprocessing(X_train)
    rf_model = train_base_model(X_train, y_train, preprocessing)
    evaluate_model(rf_model, X_test, y_test, figures_dir, name="base")
    best_model = hyperparameter_tuning(rf_model, X_train, y_train)
    evaluate_model(best_model, X_test, y_test, figures_dir, name="optimized")
    save_feature_importance(best_model, num_cols, cat_cols, figures_dir)
    save_model(best_model, model_path)

if __name__ == "__main__":
    main()