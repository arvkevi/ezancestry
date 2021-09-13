import warnings
from pathlib import Path

import joblib
import pandas as pd
from cyvcf2 import VCF
from loguru import logger
from sklearn.preprocessing import OneHotEncoder
from snps import SNPs

from ezancestry.config import aisnps_directory as _aisnps_directory
from ezancestry.config import aisnps_set as _aisnps_set
from ezancestry.config import models_directory as _models_directory
from ezancestry.config import samples_directory as _samples_directory

warnings.simplefilter(action="ignore", category=pd.errors.DtypeWarning)


def get_1kg_labels(samples_directory=None):
    """
    Get the ancestry labels for the 1000 Genomes Project samples.

    :param aisnps_directory: [description]
    :type aisnps_directory: [type]
    :return: DataFrame of sample-level population information
    :rtype: pandas DataFrame
    """
    if samples_directory is None:
        samples_directory = _samples_directory

    dfsamples = pd.read_csv(
        Path(samples_directory).joinpath(
            "integrated_call_samples_v3.20130502.ALL.panel"
        ),
        sep="\t",
    )
    dfsamples.set_index("sample", inplace=True)
    dfsamples.drop(columns=["Unnamed: 4", "Unnamed: 5"], inplace=True)
    dfsamples.columns = ["population", "superpopulation", "gender"]
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


def encode_genotypes(
    df,
    aisnps_set="Kidd",
    overwrite_encoder=False,
    models_directory=None,
    aisnps_directory=None,
):
    """One-hot encode the genotypes
    :param df: A DataFrame of samples with genotypes as columns
    :type df: pandas DataFrame
    :param aisnps_set: One of either {Kidd, Seldin}
    :type aisnps_set: str
    :param overwrite_encoder: Flag whether or not to overwrite the saved encoder for the given aisnps_set. Default: False, will load the saved encoder model.
    :type overwrite_encoder: bool
    :param models_directory: Path to the directory where the saved encoder model is saved. Default: None, will use the default location.
    :type models_directory: str
    :param aisnps_directory: Path to the directory where the AISNPs are saved. Default: None, will use the default location.
    :type aisnps_directory: str
    :return: pandas DataFrame of one-hot encoded columns for genotypes and OHE instance
    :rtype: pandas DataFrame, OneHotEncoder instance
    """

    if models_directory is None:
        models_directory = _models_directory
    if aisnps_directory is None:
        aisnps_directory = _aisnps_directory

    models_directory = Path(models_directory)
    aisnps_directory = Path(aisnps_directory)

    aisnps_set = aisnps_set.upper()
    try:
        aisnps = pd.read_csv(
            aisnps_directory.joinpath(
                f"thousand_genomes.{aisnps_set}.dataframe.csv"
            ),
            nrows=0,
            index_col=0,
        ).drop(columns=["population", "superpopulation", "gender"])
    except FileNotFoundError:
        logger.critical("""aisnps_set must be either "Kidd" or "Seldin".""")
        return

    # concact will add snps (columns) to the df that aren't in the user-submitted
    # df. Then drop the snps (columns) that are in the user-submitted df, but not
    # in the aisnps set.
    df = pd.concat([aisnps, df])[aisnps.columns]

    # TODO: Impute missing values
    # imputer = KNNImputer(n_neighbors=9)
    # imputed_aisnps = imputer.fit_transform(df)

    if overwrite_encoder:
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


def process_user_input(input_data, aisnps_directory=None, aisnps_set=None):
    """Process the user-submitted input data.

    :param input_data: [description]
    :type input_data: [type]
    :param aisnps_directory: [description], defaults to None
    :type aisnps_directory: [type], optional
    :param aisnps_set: [description], defaults to None
    :type aisnps_set: [type], optional
    :return: DataFrame where samples are row and genotypes are columns
    :rtype: pandas DataFrame
    """
    if aisnps_directory is None:
        aisnps_directory = _aisnps_directory
    if aisnps_set is None:
        aisnps_set = _aisnps_set

    aisnps_directory = Path(aisnps_directory)

    aisnpsdf = pd.read_csv(
        aisnps_directory.joinpath(f"{aisnps_set}.AISNP.txt"),
        dtype={"rsid": str, "chromosome": str, "position_hg19": int},
        sep="\t",
    )

    # If the user-submitted input ata is a directory, loop over all the files
    # to create a DataFrame of all the input data.
    if Path(input_data).is_dir():
        snpsdf = pd.DataFrame(
            columns=[
                col
                for col in aisnpsdf.columns
                if col not in ["rsid", "chromosome", "position_hg19"]
            ]
        )
        for filepath in Path(input_data).iterdir():
            try:
                snpsdf = pd.concat(
                    [snpsdf, _file_to_dataframe(filepath, aisnpsdf)]
                )
            except Exception as e:
                logger.debug(e)
                logger.warning(f"Skipping {filepath} because it was not valid")

    # The user-submitted input data is a single file.
    else:
        # _file_to_dataframe needs a Path object
        input_data = Path(input_data)
        try:
            snpsdf = _file_to_dataframe(input_data, aisnpsdf)
            # SNPs will try to read the DataFrame file
            if snpsdf is not None:
                return snpsdf
            logger.debug(
                "input_data is not a valid SNPs format, that's ok, trying to read as a pre-formatted DataFrame"
            )
        except:
            logger.debug(
                "input_data is not a valid SNPs format, that's ok, trying to read as a pre-formatted DataFrame"
            )
        try:
            snpsdf = pd.read_csv(
                input_data, index_col=0, sep=None, engine="python", dtype=str
            )
            # Need to clean up the dataframe if there is extra stuff in it
            # keep the first column, it's the index
            cols_to_keep = [snpsdf.columns[0]]
            for col in snpsdf.columns[1:]:
                if col.startswith("rs"):
                    cols_to_keep.append(col)
            return snpsdf[cols_to_keep]
        except:
            raise ValueError(
                f"{input_data} is not a valid file or directory. Please provide a valid file or directory."
            )
    return snpsdf


def _file_to_dataframe(filename, aisnpsdf):
    """Reads one file and returns a pandas DataFrame.

    :param aisnpsdf: A DataFrame of AISNPs
    :type aisnpsdf: pandas DataFrame
    :param filename: Path object to the file to be read
    :type filename: Path
    :return: A DataFrame of one record and many columns for each AISNP.
    :rtype: pandas DataFrame
    """
    # try to read a single file
    try:
        snpsobj = SNPs(str(filename))
        if snpsobj.count == 0:
            return None
        else:
            snpsdf = snpsobj.snps
        snpsdf = snpsdf.reset_index()
        snpsdf.rename(
            columns={"chrom": "chromosome", "pos": "position_hg19"},
            inplace=True,
        )
        # subset to AISNPs
        snpsdf = aisnpsdf.merge(
            snpsdf, on=["rsid", "chromosome", "position_hg19"], how="left"
        )
        # inform user how many missing snps
        n_aisnps = snpsdf["genotype"].notnull().sum()
        n_aisnps_total = snpsdf.shape[0]
        logger.info(
            f"{filename.name} sample has a valid genotype for {n_aisnps} out of a possible {n_aisnps_total} ({(n_aisnps / n_aisnps_total) * 100}%)"
        )

        snpsdfT = pd.DataFrame(columns=snpsdf["rsid"].tolist())
        snpsdfT.loc[filename.name] = snpsdf["genotype"].tolist()

        return snpsdfT

    except FileNotFoundError:
        logger.critical(f"Could not find file {filename}")

    except Exception as e:
        logger.debug(e)