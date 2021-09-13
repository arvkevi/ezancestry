from pathlib import Path

import joblib
from loguru import logger
from sklearn.neighbors import KNeighborsClassifier

from ezancestry.config import aisnps_set as _aisnps_set
from ezancestry.config import algorithm as _algorithm
from ezancestry.config import models_directory as _models_directory
from ezancestry.config import population_level as _population_level


def train(
    df,
    labels,
    algorithm=None,
    aisnps_set=None,
    k=None,
    models_directory=None,
    population_level=None,
    overwrite_model=False,
):
    """Fit and return a KNeighborsClassifier model (and optionally save it when overwite_model=True)

    :param df: Should be either the df_encoded or df_reduced DataFrmae
    :type df: pandas DataFrame
    :labels: The labels for the df
    :type labels: array, list
    :param algorithm: The dimensionality reduction algorithm that was used
    :type algorithm: str
    :param aisnps_set: The set of AISNPs to use
    :type aisnps_set: str
    :param k: The number of nearest neighbors to consider
    :type k: int
    :models_directory: The directory to save the models to
    :type models_directory: str
    :param population_level: The population level to use
    :type population_level: str
    :param overwrite_model: Whether to overwrite the model if it already exists
    :type overwrite_model: bool
    :return: The trained KNN model
    :rtype: fit KNeighborsClassifier
    """

    if k is None:
        k = _k
    if population_level is None:
        population_level = _population_level
    if aisnps_set is None:
        aisnps_set = _aisnps_set
    if algorithm is None:
        algorithm = _algorithm
    if models_directory is None:
        models_directory = _models_directory

    models_directory = Path(models_directory)

    algorithm = algorithm.upper()
    aisnps_set = aisnps_set.upper()
    population_level = (
        population_level.replace("-", "").replace(" ", "").upper()
    )

    # Create the model
    model = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=4)
    model.fit(df, labels)

    # Save the model
    if overwrite_model:
        joblib.dump(
            model,
            models_directory.joinpath(
                f"KNN.{algorithm}.{aisnps_set}.{population_level}.bin"
            ),
        )
        logger.info(f"Wrote the KNN model to: {models_directory}")

    return model


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
        knn = joblib.load(str(trained_model))
        logger.info(f"Successfully loaded trained KNN model: {trained_model}")
    except Exception as e:
        logger.error(f"Could not load the KNN model: {e}")
        knn = trained_model
        logger.info("Using user-provided KNN model")

    user_pop = knn.predict(ancestrydf)
    user_pop_probs = knn.predict_proba(ancestrydf)

    ancestrydf["predicted_population"] = user_pop
    ancestrydf[knn.classes_] = user_pop_probs

    return ancestrydf
