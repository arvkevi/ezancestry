from pathlib import Path

import joblib
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from cyvcf2 import VCF
from loguru import logger

from ezancestry.settings import (
    models_directory,
    samples_directory,
    aisnps_directory,
)


def get_1kg_labels():
    """
    Get the ancestry labels for the 1000 Genomes Project samples.

    :param aisnps_directory: [description]
    :type aisnps_directory: [type]
    :return: DataFrame of sample-level population information
    :rtype: pandas DataFrame
    """
    dfsamples = pd.read_csv(
        Path(samples_directory).joinpath(
            "integrated_call_samples_v3.20130502.ALL.panel"
        ),
        sep="\t",
    )
    dfsamples.set_index("sample", inplace=True)
    dfsamples.drop(columns=["Unnamed: 4", "Unnamed: 5"], inplace=True)
    dfsamples.columns = ["population", "super population", "gender"]
    return dfsamples


def vcf2df(vcf_fname, dfsamples):
    """Convert a vcf file (from the 1kg AISNPs) to a pandas DataFrame
    :param vcf_fname: path to the vcf file with AISNPs for every 1kg sample
    :type vcf_fname: str
    :param dfsamples: DataFrame with sample-level info on each 1kg sample.
    :type dfsamples: pandas DataFrame
    :return: DataFrame with genotypes for AISNPs as columns and samples as rows.
    :rtype: pandas DataFrame
    """
    vcf_file = VCF(vcf_fname)
    df = pd.DataFrame(index=vcf_file.samples)
    for variant in vcf_file():
        # TODO: ensure un-phasing variants is the desired behavior
        # sorted() normalizes the order of the genotypes
        df[variant.ID] = [
            "".join(sorted(gt.replace("|", ""))) for gt in variant.gt_bases
        ]

    df = df.join(dfsamples, how="inner")

    return df


def encode_genotypes(df, aisnps_set="Kidd", overwrite_encoder=False):
    """One-hot encode the genotypes
    :param df: A DataFrame of samples with genotypes as columns
    :type df: pandas DataFrame
    :param aisnps_set: One of either {Kidd, Seldin}
    :type aisnps_set: str
    :param overwrite_encoder: Flag whether or not to overwrite the saved encoder for the given aisnps_set. Default: False, will load the saved encoder model.
    :type overwrite_encoder: bool
    :return: pandas DataFrame of one-hot encoded columns for genotypes and OHE instance
    :rtype: pandas DataFrame, OneHotEncoder instance
    """
    aisnps_set = aisnps_set.upper()
    try:
        aisnps = pd.read_csv(
            aisnps_directory.joinpath(f"thousand_genomes.{aisnps_set}.dataframe.csv"),
            nrows=0,
            index_col=0,
        ).drop(columns=["population", "super population", "gender"])
    except FileNotFoundError:
        logger.critical("""aisnps_set must be either "Kidd" or "Seldin".""")
        return

    # concact will add snps (columns) to the df that aren't in the user-submitted
    # df. Then drop the snps (columns) that are in the user-submitted df, but not
    # in the aisnps set.
    df = pd.concat([aisnps, df])[aisnps.columns]

    if overwrite_encoder:
        # TODO: handle_unknown sets novel genotypes to all zeros for each category.
        # 1. Use a different encoder technique (OHE works for now)
        # 2. Pass a list of valid genotypes (overkill, dimensionality explodes)
        ohe = OneHotEncoder(
            sparse=False,
            handle_unknown="ignore",
        )
        X = ohe.fit_transform(df.values)
        # overwrite the old encoder with a new one
        joblib.dump(
            ohe, models_directory.joinpath(f"one_hot_encoder.{aisnps_set}.bin")
        )
        logger.info(
            f"Wrote a new encoder to {models_directory}/one_hot_encoder.{aisnps_set}.bin"
        )
    else:
        ohe = joblib.load(
            models_directory.joinpath(f"one_hot_encoder.{aisnps_set}.bin")
        )
        logger.info(
            f"Successfully loaded an encoder from {models_directory}/one_hot_encoder.{aisnps_set}.bin"
        )
        X = ohe.transform(df.values)

    return pd.DataFrame(
        X, index=df.index, columns=ohe.get_feature_names(df.columns.tolist())
    )
