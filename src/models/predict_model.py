import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


class BatchPredictor:
    def __init__(self):
        self.scaler = None

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at: {model_path}"
            )
        return joblib.load(model_path)

    def load_data(self, input_path):
        if not os.path.exists(input_path):
            raise FileNotFoundError(
                f"Input file not found at: {input_path}"
            )
        return pd.read_csv(input_path)

    def preprocess(self, df):
        numeric_cols = df.select_dtypes(
            include=["int64", "float64"]
        ).columns
        self.scaler = StandardScaler()
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df

    def predict(self, df, model):
        preds = model.predict(df)
        return preds

    def save_predictions(self, df, preds, output_path):
        df["prediction"] = preds
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    def predict_file(self, input_path, model_path, output_path):
        df = self.load_data(input_path)
        df_processed = self.preprocess(df)
        model = self.load_model(model_path)
        preds = self.predict(df_processed, model)
        self.save_predictions(df, preds, output_path)
