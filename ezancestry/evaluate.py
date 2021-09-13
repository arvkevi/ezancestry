from pathlib import Path
from typing import Union

import joblib
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate

from ezancestry.aisnps import extract_aisnps
from ezancestry.config import aisnps_directory as _aisnps_directory
from ezancestry.config import aisnps_set as _aisnps_set
from ezancestry.config import algorithm as _algorithm
from ezancestry.config import k as _k
from ezancestry.config import models_directory as _models_directory
from ezancestry.config import n_components as _n_components
from ezancestry.config import population_level as _population_level
from ezancestry.config import samples_directory as _samples_directory
from ezancestry.config import \
    thousand_genomes_directory as _thousand_genomes_directory
from ezancestry.dimred import dimensionality_reduction
from ezancestry.fetch import download_thousand_genomes
from ezancestry.model import predict_ancestry, train
from ezancestry.process import encode_genotypes, get_1kg_labels, vcf2df


def export_performance(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: Union[pd.Series, pd.DataFrame],
    y_test: Union[pd.Series, pd.DataFrame],
    models_directory: str = None,
    aisnps_directory: str = None,
    population_level: str = None,
    algorithm: str = None,
    aisnps_set: str = None,
    outdir: str = None,
):
    """Evaluate the performance of the pre-trained model. For this function to
     work as intended, the models must have already been fit on df_train.

    :param df_train: DataFrame of reduced dimensionality training data.
    :type df_train: pd.DataFrame
    :param df_test: DataFrame of reduced dimensionality test data.
    :type df_test: pd.DataFrame
    :param models_directory: [description], defaults to None
    :type models_directory: str, optional
    :param aisnps_directory: [description], defaults to None
    :type aisnps_directory: str, optional
    :param population_level: [description], defaults to None
    :type population_level: str, optional
    :param algorithm: [description], defaults to None
    :type algorithm: str, optional
    :param n_components: [description], defaults to None
    :type n_components: int, optional
    :param k: [description], defaults to None
    :type k: int, optional
    :param thousand_genomes_directory: [description], defaults to None
    :type thousand_genomes_directory: str, optional
    :param samples_directory: [description], defaults to None
    :type samples_directory: str, optional
    :param aisnps_set: [description], defaults to None
    :type aisnps_set: str, optional
    :param outdir: Where to write the result files, defaults to models_directory
    :type outdir: str, optional
    """

    if models_directory is None:
        models_directory = _models_directory
    if aisnps_directory is None:
        aisnps_directory = _aisnps_directory
    if population_level is None:
        population_level = _population_level
    if algorithm is None:
        algorithm = _algorithm
    if aisnps_set is None:
        aisnps_set = _aisnps_set
    if outdir is None:
        outdir = models_directory

    models_directory = Path(models_directory)
    aisnps_directory = Path(aisnps_directory)
    outdir = Path(outdir)

    population_level = (
        population_level.replace("-", "").replace(" ", "").upper()
    )
    algorithm = algorithm.upper()
    aisnps_set = aisnps_set.upper()

    # Load the pre-trained model
    model = models_directory.joinpath(
        f"knn.{algorithm}.{aisnps_set}.{population_level}.bin"
    )
    model = joblib.load(model)
    logger.info("Predicting ancestry for evaluation...")
    dftrain_results = predict_ancestry(df_train, model)
    dftest_results = predict_ancestry(df_test, model)

    # Perform cross validation
    logger.info("Performing cross validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model,
        df_train,
        y_train,
        cv=cv,
        scoring=["f1_macro", "precision_macro", "recall_macro", "accuracy"],
        return_train_score=True,
        return_estimator=True,
    )
    # Save the CV results
    cv_results_file = outdir.joinpath(
        f"knn.{algorithm}.{aisnps_set}.{population_level}.cv.csv"
    )
    logger.info(f"Writing cross validation results to {cv_results_file}...")
    cv_results = pd.DataFrame(cv_results)
    cv_results.index = [f"Fold {i}" for i in range(1, 6)]
    cv_results["training_accuracy"] = accuracy_score(
        y_train, model.predict(df_train)
    )
    cv_results.to_csv(cv_results_file, index=True)

    # Evaluate on the held out test set
    logger.info("Evaluating on the held out test set...")
    y_pred = dftest_results["predicted_population"]

    holdout_results = pd.DataFrame(
        classification_report(
            y_test, y_pred, target_names=model.classes_, output_dict=True
        )
    )
    holdout_results_file = outdir.joinpath(
        f"knn.{algorithm}.{aisnps_set}.{population_level}.holdout.csv"
    )
    logger.info(
        f"Writing holdout validation results to {holdout_results_file}..."
    )
    holdout_results.to_csv(holdout_results_file, index=True)

    return cv_results, holdout_results
