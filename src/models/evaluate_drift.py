import click
import pandas as pd
import seaborn as sns
from alibi_detect.cd import KSDrift
import matplotlib.pyplot as plt
from src.models.train_model import (
    DataRepository,
    DataSplitter
)


def create_kde_plot(X_train: pd.DataFrame, X_eval: pd.DataFrame, feature: str, figures_dir: str):
    """
    Inputs:
        - X_train: DataFrame with train data
        - synthetic_data: Synthetic train data
        - feature: Column to analyze
        - figures_dir: Path to store figures
    Purpose:
        - Elaborate a kde plot to observe distribution drift
    """
    plt.figure(figsize=(10,5))

    sns.kdeplot(X_train[feature], label='Original Data', fill=True, color='blue')

    sns.kdeplot(X_eval[feature], label='Eval Data', fill=True, color='red')

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
@click.argument("figures_dir", type=click.Path())
def main(
    train_data_path: str,
    eval_data_path: str,
    feature: str,
    figures_dir: str
):
    """
    Inputs:
        - train_data_path: Train dataset path
        - eval_data_path: Eval dataset path
    Purpose:
        - Calculate drift based on KSDrift method
    """

    # Read training and test data
    train = DataRepository().load(train_data_path)
    eval = DataRepository().load(eval_data_path)
    print(
        f"""\nTrain ({train_data_path})
and Eval ({eval_data_path}) data loaded\n"""
    )

    # Split data accordingly
    X_train, X_test, y_train, y_test = \
        DataSplitter().split(train)
    print(f'Train ({train_data_path}) data splitted\n')

    # Split test data into X and y
    X_eval = eval[X_train.columns]

    # Train KSDrift and infer on eval data
    print('Calculating KSDrift\n')
    cd = KSDrift(X_train.values, p_val=0.05)
    drift_pred = cd.predict(X_eval.values)

    print('SK-Drifted detected: ', bool(drift_pred['data']['is_drift']), '\n')

    create_kde_plot(X_train, X_eval, feature, figures_dir)
    print('kde plot generated')

if __name__ == "__main__":
    main()
