import numpy as np
import pandas as pd
import umap

from category_encoders.one_hot import OneHotEncoder
from cyvcf2 import VCF
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer


def get_file_content_as_string(mdfile):
    """Convenience function to convert file to string

    :param mdfile: path to markdown
    :type mdfile: str
    :return: file contents
    :rtype: str
    """
    mdstring = ""
    with open(mdfile, "r") as f:
        for line in f:
            mdstring += line
    return mdstring


def get_1kg_samples(onekg_samples="data/integrated_call_samples_v3.20130502.ALL.panel"):
    """Download the sample information for the 1000 Genomes Project

    :return: DataFrame of sample-level population information
    :rtype: pandas DataFrame
    """
    dfsamples = pd.read_csv(onekg_samples, sep="\t")
    dfsamples.set_index("sample", inplace=True)
    dfsamples.drop(columns=["Unnamed: 4", "Unnamed: 5"], inplace=True)
    dfsamples.columns = ["population", "super population", "gender"]
    return dfsamples


def encode_genotypes(df):
    """One-hot encode the genotypes

    :param df: A DataFrame of samples with genotypes as columns
    :type df: pandas DataFrame
    :return: pandas DataFrame of one-hot encoded columns for genotypes and OHE instance
    :rtype: pandas DataFrame, OneHotEncoder instance
    """
    ohe = OneHotEncoder(cols=df.columns, handle_missing="return_nan")
    X = ohe.fit_transform(df)
    return pd.DataFrame(X, index=df.index), ohe


def dimensionality_reduction(X, algorithm="PCA"):
    """Reduce the dimensionality of the AISNPs
    :param X: One-hot encoded 1kG AISNPs.
    :type X: pandas DataFrame
    :param algorithm: The type of dimensionality reduction to perform.
        One of {PCA, UMAP, t-SNE}
    :type algorithm: str
    :returns: The transformed X DataFrame, reduced to 3 components by <algorithm>,
    and the dimensionality reduction Transformer object.
    """
    n_components = 3

    if algorithm == "PCA":
        reducer = PCA(n_components=n_components)
    elif algorithm == "t-SNE":
        reducer = TSNE(n_components=n_components, n_jobs=4)
    elif algorithm == "UMAP":
        reducer = umap.UMAP(
            n_components=n_components, min_dist=0.2, metric="dice", random_state=42
        )
    else:
        return None, None

    X_reduced = reducer.fit_transform(X.values)

    return pd.DataFrame(X_reduced, columns=["x", "y", "z"], index=X.index), reducer


def filter_user_genotypes(userdf, aisnps_1kg):
    """Filter the user's uploaded genotypes to the AISNPs

    :param userdf: The user's DataFrame from SNPs
    :type userdf: pandas DataFrame
    :param aisnps_1kg: The DataFrame containing snps for the 1kg project samples
    :type aisnps_1kg: pandas DataFrame
    :return: The user's DataFrame of AISNPs as columns, The 1kg DataFrame with user appended
    :rtype: pandas DataFrame
    """
    user_record = pd.DataFrame(index=["your_sample"], columns=aisnps_1kg.columns)
    for snp in user_record.columns:
        try:
            user_record[snp] = userdf.loc[snp]["genotype"]
        except KeyError:
            continue
    aisnps_1kg = aisnps_1kg.append(user_record)
    return user_record, aisnps_1kg


def impute_missing(aisnps_1kg):
    """Use scikit-learns KNNImputer to impute missing genotypes for AISNPs

    :param aisnps_1kg: DataFrame of all samples including user's encoded genotypes.
    :type aisnps_1kg: pandas DataFrame
    :return: DataFrame with nan values filled in my KNNImputer
    :rtype: pandas DataFrame
    """
    imputer = KNNImputer(n_neighbors=9)
    imputed_aisnps = imputer.fit_transform(aisnps_1kg)
    return np.rint(imputed_aisnps[-1])


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

    df = df.join(dfsamples, how="outer")

    return df