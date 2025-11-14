import click
import pandas as pd
import seaborn as sns
from alibi_detect.cd import KSDrift
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from src.models.train_model import (
    DataRepository,
    DataSplitter
)


def create_kde_plot(X_train: pd.DataFrame, synthetic_data: pd.DataFrame, feature: str, figures_dir: str):
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

    sns.kdeplot(synthetic_data[feature], label='Shifted Data', fill=True, color='red')

    plt.title('Comparison Original vs Drifted', fontsize=16)
    plt.xlabel(feature, fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{figures_dir}/kde_plot_simulate_{feature}.png')
    plt.close()


@click.command()
@click.argument("train_data_path", type=click.Path(exists=True))
@click.argument("n_components", type=click.INT)
@click.argument("feature", type=click.STRING)
@click.argument("meanshift", type=click.FLOAT)
@click.argument("figures_dir", type=click.Path())
def main(
    train_data_path: str,
    n_components: int,
    feature: str,
    meanshift: float,
    figures_dir: str
):
    """
    Inputs:
        - train_data_path: Train dataset path
        - n_components: Components for Gaussian Mixture
        - feature: Column name for evaluation
        - meanshift: Number to modify means
    Purpose:
        - Calculate drift based on KSDrift method
    """

    # Read training and test data
    train = DataRepository().load(train_data_path)
    print(f'\nTrain ({train_data_path})data loaded\n')

    # Split data accordingly
    X_train, X_test, y_train, y_test = \
        DataSplitter().split(train)
    print(f'Train ({train_data_path}) data splitted\n')

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
    print('\nModified Means: \n', mod_means)

    # Generate synthetic data with shifted means
    synthetic_data = X_train.copy()

    # Save synthetic data set
    synthetic_data.to_csv('data/clean/steel_energy_synthetic.csv')

    synthetic_data[feature] = gmm.sample(
        int(X_train.describe()[feature]['count'])
    )[0].reshape(-1)
    X_synthetic = synthetic_data[X_train.columns].values

    # Train KSDrift and infer on eval data
    print('Calculating KSDrift\n')
    cd = KSDrift(X_train.values, p_val=0.05)
    drift_pred = cd.predict(X_synthetic)

    print('SK-Drifted detected: ', bool(drift_pred['data']['is_drift']), '\n')

    create_kde_plot(X_train, synthetic_data, feature, figures_dir)
    print('kde plot generated')


if __name__ == "__main__":
    main()
