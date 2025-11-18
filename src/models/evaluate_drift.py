import json
import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from alibi_detect.cd import KSDrift
from sklearn.mixture import GaussianMixture
from src.models.predict_model import BatchPredictor
from src.models.train_model import (
    DataRepository,
    DataSplitter
)


def feature_pipeline_healthcheck(
    is_drifted: bool,
    feature: str
) -> bool:
    """
    Arguments:
        is_drifted: Boolean indicating if there is drift
        feature: Feature evaluated
    Returns:
        healthy: boolean check
    """
    # If is drifted, then look into feature pipeline
    if is_drifted:
        status = f'Check feature {feature}'
        healthy = False
    else:
        status = 'Healthy'
        healthy = True

    # Print status
    print(f'Feature Pipeline Status: {status}\n')

    return healthy


def model_health_check(
    metrics_path_train: str,
    metrics_path_eval: str,
    metric: str = 'accuracy',
    threshold: float = 0.05
) -> bool:
    """
    Arguments:
        metrics_path_train: json with train metrics
        metrics_path_eval: json with eval metrics
        metric: metric to evaluate
        threshold: accepted difference
    Returns:
        healthy: boolean check
    """
    # Read metrics json files
    with open(metrics_path_train, 'r') as f:
        train_metrics = json.load(f)

    with open(metrics_path_eval, 'r') as f:
        eval_metrics = json.load(f)

    # Get metric to evaluate
    train_metric = train_metrics[metric]
    eval_metric = eval_metrics[metric]

    # Calculate absolute diference
    abs_diff = abs(train_metric - eval_metric)

    # Determine model health
    if abs_diff > threshold:
        status = 'Check model, possible retraining'
        healthy = False
    else:
        status = 'Healthy'
        healthy = True

    # Print status
    print(f'Model Status: {status}\n')

    return healthy


def generate_drifted_dataset(
    df: pd.DataFrame,
    feature: str,
    n_components: int,
    meanshift: float
) -> tuple[np.ndarray, str]:
    """
    Arguments:
        df: DataFrame with train data
        feature: Column to analyze
        n_components: Components for GaussianMixture
        meanshift: Factor to which shift means
    Returns:
        X_synthetic: Artifitial dataset for drifting
        synthetic_data_path: Path to artifitial dataset
    """
    # Copy data so we dont affect original data
    X_train = df.copy()

    # Split test data into X and y
    gmm = GaussianMixture(
        n_components=n_components,
        random_state=42
    ).fit(
        X_train[feature].to_numpy().reshape(-1, 1)
    )
    print('\nOriginal Means: \n', gmm.means_)

    # Modify the means
    mod_means = gmm.means_.copy()
    for i in range(n_components):
        mod_means[i] *= (1 + meanshift)

    # Assign the modified means back to gmm
    gmm.means_ = mod_means
    print('\nModified Means: \n', mod_means, '\n')

    # Generate synthetic data with shifted means
    synthetic_data = X_train.copy()

    synthetic_data[feature] = gmm.sample(
        int(X_train.describe()[feature]['count'])
    )[0].reshape(-1)

    # Save synthetic data set
    synthetic_data_path = 'data/clean/steel_energy_synthetic.csv'
    synthetic_data.to_csv(synthetic_data_path)

    return synthetic_data_path


def create_kde_plot(
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    feature: str,
    figures_dir: str
):
    """
    Inputs:
        - X_train: DataFrame with train data
        - synthetic_data: Synthetic train data
        - feature: Column to analyze
        - figures_dir: Path to store figures
    Purpose:
        - Elaborate a kde plot to observe distribution drift
    """
    plt.figure(figsize=(10, 5))

    sns.kdeplot(
        X_train[feature],
        label='Original Data',
        fill=True,
        color='blue'
    )

    sns.kdeplot(
        X_eval[feature],
        label='Eval Data',
        fill=True,
        color='red'
    )

    plt.title('Comparison Original vs Evaluation', fontsize=16)
    plt.xlabel(feature, fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{figures_dir}/kde_plot_evaluate_{feature}.png')
    plt.close()


@click.command()
@click.argument("train_data_path", type=click.Path(exists=True))
@click.argument("eval_data_path", type=click.Path())
@click.argument("feature", type=click.Path())
@click.argument("model_path", type=click.Path())
@click.argument("reports_dir", type=click.Path())
@click.argument("simulate_drift", required=False,
                default=False, type=click.BOOL)
@click.argument("n_components", required=False, default=5, type=click.INT)
@click.argument("meanshift", required=False, default=0.3, type=click.FLOAT)
@click.argument("metric", required=False,
                default='accuracy', type=click.STRING)
@click.argument("threshold", required=False, default=0.05, type=click.FLOAT)
def main(
    train_data_path: str,
    eval_data_path: str,
    feature: str,
    model_path: str,
    reports_dir: str,
    simulate_drift: bool = False,
    n_components: int = 5,
    meanshift: float = 0.3,
    metric: str = 'accuracy',
    threshold: float = 0.05
):
    """
    Arguments:
        - train_data_path: Train dataset path
        - eval_data_path: Eval dataset path
    Purpose:
        - Calculate drift based on KSDrift method
    """

    # Read training and test data
    train = DataRepository().load(train_data_path)
    print(f'\nTrain data ({train_data_path}) loaded\n')

    # Split data accordingly
    X_train, X_test, y_train, y_test = \
        DataSplitter().split(train)
    print(f'Train ({train_data_path}) data splitted\n')

    # In case we want to simulate drift
    if simulate_drift:
        print(f'Generating Synthetic Dataset based on {train_data_path}')
        eval_data_path = generate_drifted_dataset(
            X_train,
            feature,
            n_components,
            meanshift
        )

    eval = DataRepository().load(eval_data_path)
    print(f'Eval data ({eval_data_path}) loaded\n')

    X_eval = eval[X_train.columns]

    # Train KSDrift and infer on eval data
    print('Calculating KSDrift\n')
    cd = KSDrift(X_train.values, p_val=0.05)
    drift_pred = cd.predict(X_eval.values)

    is_drifted = bool(drift_pred['data']['is_drift'])
    print('SK-Drifted detected: ', is_drifted, '\n')

    create_kde_plot(X_train, X_eval, feature, reports_dir)
    print('kde plot generated\n')

    # Generate paths for metrics and predictions
    predictions_path_train = reports_dir + 'predictions_train.csv'
    predictions_path_eval = reports_dir + 'predictions_eval.csv'
    metrics_path_train = reports_dir + 'metrics_train.json'
    metrics_path_eval = reports_dir + 'metrics_eval.json'

    # Generate predictions on pretrain model
    BatchPredictor().predict_file(
        train_data_path,
        model_path,
        predictions_path_train,
        metrics_path_train,
        reports_dir
    )
    print('Predictions and Metrics for Training Data\n')

    BatchPredictor().predict_file(
        eval_data_path,
        model_path,
        predictions_path_eval,
        metrics_path_eval,
        reports_dir
    )
    print('Predictions and Metrics for Eval Data\n')

    # Check Feature Pipeline health
    feature_pipeline_healthcheck(
        is_drifted,
        feature
    )

    # Check Model health
    model_health_check(
        metrics_path_train,
        metrics_path_eval,
        metric,
        threshold
    )


if __name__ == "__main__":
    main()
