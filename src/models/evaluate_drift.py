import click
import pandas as pd
from alibi_detect.cd import KSDrift
from src.models.train_model import (
    DataRepository,
    DataSplitter,
    PreprocessingBuilder
)


@click.command()
@click.argument("train_data_path", type=click.Path(exists=True))
@click.argument("eval_data_path", type=click.Path())
def main(train_data_path: str, eval_data_path: str) -> bool:
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

    # Preprocess data as in training phase
    pipeline, cat_cols, num_cols = PreprocessingBuilder().build(X_train)
    X_train_proc = pd.DataFrame(
        pipeline.fit_transform(X_train),
        columns=pipeline.get_feature_names_out()
    )
    print(f'Train ({train_data_path}) data preprocessed\n')

    # Split test data into X and y
    X_eval = eval[X_train.columns]

    # Preprocess the eval dataset as with the training data
    X_eval_proc = pd.DataFrame(
        pipeline.transform(X_eval),
        columns=pipeline.get_feature_names_out()
    )
    print(f'Eval ({eval_data_path}) data preproccesed\n')

    # Train KSDrift and infer on eval data
    print('Calculating KSDrift\n')
    cd = KSDrift(X_train_proc.values, p_val=0.05)
    drift_pred = cd.predict(X_eval_proc.values)

    print('SK-Drifted detected: ', bool(drift_pred['data']['is_drift']), '\n')


if __name__ == "__main__":
    main()
