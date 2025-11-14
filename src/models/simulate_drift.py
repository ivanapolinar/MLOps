import click
import pandas as pd
from alibi_detect.cd import KSDrift
from sklearn.mixture import GaussianMixture
from src.models.train_model import (
    DataRepository,
    DataSplitter,
    PreprocessingBuilder
)


@click.command()
@click.argument("train_data_path", type=click.Path(exists=True))
@click.argument("n_components", type=click.INT)
@click.argument("feature", type=click.STRING)
@click.argument("meanshift", type=click.FLOAT)
def main(
    train_data_path: str,
    n_components: int,
    feature: str,
    meanshift: float
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

    # Preprocess data as in training phase
    pipeline, cat_cols, num_cols = PreprocessingBuilder().build(X_train)
    X_train_proc = pd.DataFrame(
        pipeline.fit_transform(X_train),
        columns=pipeline.get_feature_names_out()
    )
    print(f'Train ({train_data_path}) data preprocessed\n')

    # Split test data into X and y
    gmm = GaussianMixture(
        n_components=n_components,
        random_state=42
    ).fit(
        X_train_proc[feature].to_numpy().reshape(-1, 1)
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
    synthetic_data = X_train_proc.copy()
    synthetic_data[feature] = gmm.sample(
        int(X_train_proc.describe()[feature]['count'])
    )[0].reshape(-1)
    X_synthetic = synthetic_data[X_train_proc.columns].values

    # Train KSDrift and infer on eval data
    print('Calculating KSDrift\n')
    cd = KSDrift(X_train_proc.values, p_val=0.05)
    drift_pred = cd.predict(X_synthetic)

    print('SK-Drifted detected: ', bool(drift_pred['data']['is_drift']), '\n')


if __name__ == "__main__":
    main()
