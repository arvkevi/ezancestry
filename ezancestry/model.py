from pathlib import Path

import joblib
from loguru import logger

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

from ezancestry.config import aisnps_set as _aisnps_set
from ezancestry.config import models_directory as _models_directory
from ezancestry.config import population_level as _population_level

import numpy as np


DEFAULT_PIPELINE = make_pipeline(
    OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.int8),
    KNNImputer(n_neighbors=7),
    PCA(n_components=10),
    KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=4),
)


def train(
    df,
    labels,
    sklearn_pipeline=None,
    aisnps_set=None,
    models_directory=None,
    population_level=None,
    overwrite_model=False,
):
    """Fit and return a pipeline model (and optionally save it when overwite_model=True)

    :param df: Should be either the df_encoded or df_reduced DataFrmae
    :type df: pandas DataFrame
    :labels: The labels for the df
    :type labels: pandas Series
    :param sklearn_pipeline: The pipeline to use for training the model
    :type sklearn_pipeline: sklearn.pipeline.Pipeline
        
        If you want to use a custom pipeline, you can pass it here. Otherwise, the default pipeline will be used:
        
        DEFAULT_PIPELINE = make_pipeline(
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            KNNImputer(n_neighbors=7),
            PCA(n_components=3),
            KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=4),
        )

    :param aisnps_set: The aisnps_set to use for training the model
    :type aisnps_set: str
    :param models_directory: The directory to save the model to
    :type models_directory: str
    :param population_level: The population_level to use for training the model
    :type population_level: str
    :param overwrite_model: Whether to overwrite the model
    :type overwrite_model: bool
        
        The default is False
        When True, the model will be saved to the models_directory as a .bin file
        with the name <aisnps_set>.<population_level>.bin

    :return: The trained model
    :rtype: sklearn.pipeline.Pipeline
    """

    if population_level is None:
        population_level = _population_level
    if aisnps_set is None:
        aisnps_set = _aisnps_set
    if models_directory is None:
        models_directory = _models_directory

    models_directory = Path(models_directory)
    aisnps_set = aisnps_set.lower()
    population_level = (
        population_level.replace("-", "").replace(" ", "").lower()
    )

    if population_level not in ["population", "superpopulation"]:
        raise ValueError(
            "population_level must be either 'population' or 'superpopulation'"
        )

    # Create the model
    if sklearn_pipeline is None:
        sklearn_pipeline = DEFAULT_PIPELINE

    sklearn_pipeline.fit(df, labels)

    # Save the model
    if overwrite_model:
        joblib.dump(
            sklearn_pipeline,
            models_directory.joinpath(
                f"{aisnps_set}.{population_level}.bin"
            ),
        )
        logger.info(f"Wrote the scikit-learn pipeline to: {models_directory}")

    return sklearn_pipeline


def predict_ancestry(df, trained_model):
    """Predict the ancestry for a given DataFrame and return it

    :param df: The df_encoded or df_reduced DataFrame to predict on.
    :type df: pandas DataFrame
    :param trained_model: Path to the trained model, or the model itself
    :type trained_model: str or fit KNeighborsClassifier
    :return: A dataframe with the predictions
    :rtype: pandas DataFrame
    """
    ancestrydf = df.copy()
    try:
        model = joblib.load(str(trained_model))
        logger.info(f"Successfully loaded trained model: {trained_model}")
    except Exception as e:
        logger.warning(f"Could not load the model: {e}")
        model = trained_model
        logger.info("Using user-provided model")

    user_pop = model.predict(ancestrydf)
    user_pop_probs = model.predict_proba(ancestrydf)

    ancestrydf["predicted_population"] = user_pop
    ancestrydf[model.classes_] = user_pop_probs

    return ancestrydf
