import click
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


class BatchPredictor:
    def __init__(self):
        self.scaler = StandardScaler()

    def load_model(self, model_path):
        """Load trained model."""
        return joblib.load(model_path)

    def preprocess(self, df):
        """Preprocess input dataframe."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df

    def predict_df(self, model, df):
        """Predict directly from a dataframe."""
        processed_df = self.preprocess(df)
        predictions = model.predict(processed_df)
        return predictions

    def predict_file(self, input_path, model_path, output_path):
        """Predict from CSV file and save results."""
        df = pd.read_csv(input_path)
        model = self.load_model(model_path)

        preds = self.predict_df(model, df)
        df["prediction"] = preds

        df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")


@click.command()
@click.argument("input_path")
@click.argument("model_path")
@click.argument("output_path")
def main(input_path, model_path, output_path):
    predictor = BatchPredictor()
    predictor.predict_file(input_path, model_path, output_path)


if __name__ == "__main__":
    main()
