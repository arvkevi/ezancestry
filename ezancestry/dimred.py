import joblib
import pandas as pd
import umap
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ezancestry.settings import models_directory


def dimensionality_reduction(df, algorithm="PCA", refit=False):
    """Reduce the dimensionality of the AISNPs
    :param df: One-hot encoded 1kG AISNPs.
    :type df: pandas DataFrame
    :param algorithm: The type of dimensionality reduction to perform.
        One of {PCA, UMAP, t-SNE}
    :type algorithm: str
    :param refit:
    :type refit: bool
    :returns: The transformed X DataFrame, reduced to 3 components by <algorithm>.
    """
    algorithm = algorithm.upper()
    n_components = 3
    columns = [
        col
        for col in df.columns
        if col not in ("gender", "population", "super population")
    ]
    if refit:
        if algorithm == "PCA":
            reducer = PCA(n_components=n_components)
            joblib.dump(reducer, models_directory.joinpath("pca.bin"))
        elif algorithm == "T-SNE" or algorithm == "TSNE":
            reducer = TSNE(n_components=n_components, n_jobs=4)
            joblib.dump(reducer, models_directory.joinpath("tsne.bin"))
        elif algorithm == "UMAP":
            reducer = umap.UMAP(
                n_components=n_components,
                min_dist=0.2,
                metric="dice",
                random_state=42,
            )
            joblib.dump(reducer, models_directory.joinpath("pca.bin"))
        else:
            return None
        logger.info(
            f"Successfully write a dimensionality reduction model to {models_directory}"
        )
    else:
        # load reducer and return reduced X
        if algorithm == "PCA":
            reducer = joblib.load(models_directory.joinpath("pca.bin"))
        elif algorithm == "T-SNE" or algorithm == "TSNE":
            reducer = joblib.load(models_directory.joinpath("tsne.bin"))
        elif algorithm == "UMAP":
            reducer = joblib.load(models_directory.joinpath("umap.bin"))
        else:
            return None
        logger.info("Successfully loaded a dimensionality reduction model")
    df_reduced = reducer.fit_transform(df[columns].values)
    df_reduced = df_reduced

    return pd.DataFrame(
        df_reduced, columns=["x", "y", "z"], index=df.index
    ).join(df[["gender", "population", "super population"]], how="inner")
