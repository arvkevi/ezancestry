from pathlib import Path

import joblib
import pandas as pd
from category_encoders.one_hot import OneHotEncoder
from cyvcf2 import VCF
from loguru import logger

from ezancestry.settings import models_directory, samples_directory


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
        df[variant.ID] = [gt.replace("|", "") for gt in variant.gt_bases]

    df = df.join(dfsamples, how="inner")

    return df


def encode_genotypes(df, refit=False):
    """One-hot encode the genotypes
    :param df: A DataFrame of samples with genotypes as columns
    :type df: pandas DataFrame
    :param refit: Flag whether or not to refit the encoder.
    :type refit: bool
    :return: pandas DataFrame of one-hot encoded columns for genotypes and OHE instance
    :rtype: pandas DataFrame, OneHotEncoder instance
    """
    columns = [
        col
        for col in df.columns
        if col not in ("gender", "population", "super population")
    ]
    if refit:
        ohe = OneHotEncoder(
            cols=columns, handle_missing="return_nan", use_cat_names=True
        )
        X = ohe.fit_transform(df)
        # overwrite the old encoder with the new one
        joblib.dump(ohe, models_directory.joinpath("one_hot_encoder.bin"))
        logger.info(
            "Wrote a new encoder to models_directory/one_hot_encoder.bin"
        )
    else:
        ohe = joblib.load(models_directory.joinpath("one_hot_encoder.bin"))
        logger.info(
            "Successfully loaded an encoder from models_directory/one_hot_encoder.bin"
        )
        X = ohe.fit_transform(df)

    return pd.DataFrame(X, index=df.index)
