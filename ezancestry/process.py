import warnings
from pathlib import Path

import pandas as pd
from loguru import logger
from snps import SNPs

from ezancestry.config import aisnps_directory as _aisnps_directory
from ezancestry.config import aisnps_set as _aisnps_set

warnings.simplefilter(action="ignore", category=pd.errors.DtypeWarning)


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
        aisnps_directory.joinpath(f"{aisnps_set}.aisnp.txt"),
        dtype={"rsid": str, "chromosome": str, "position": int},
        sep="\t",
    )

    try:
        input_data_is_pathlike = bool(Path(input_data))
    except TypeError:
        input_data_is_pathlike = False

    # If the user-submitted input data is a directory, loop over all the files
    # to create a DataFrame of all the input data.
    if input_data_is_pathlike:
        if Path(input_data).is_dir():
            snpsdf = pd.DataFrame(
                columns=[
                    col
                    for col in aisnpsdf.columns
                    if col not in ["rsid", "chromosome", "position"]
                ]
            )
            for filepath in Path(input_data).iterdir():
                try:
                    snpsdf = pd.concat(
                        [snpsdf, _input_to_dataframe(filepath, aisnpsdf)]
                    )
                except Exception as e:
                    logger.debug(e)
                    logger.warning(
                        f"Skipping {filepath} because it was not valid"
                    )
            for col in snpsdf.columns:
                snpsdf[col] = snpsdf[col].astype("object")
            return snpsdf

        # The user-submitted input data is a single file.
        else:
            # _input_to_dataframe needs a Path object
            input_data = Path(input_data)
            try:
                snpsdf = _input_to_dataframe(input_data, aisnpsdf)
                # SNPs will try to read the DataFrame file
                if snpsdf is not None:
                    for col in snpsdf.columns:
                        snpsdf[col] = snpsdf[col].astype("object")
                    return snpsdf
                logger.debug(
                    "input_data is not a valid SNPs format, that's ok, trying to read as a pre-formatted DataFrame"
                )
            except Exception as e:
                logger.debug(e)
            # read the user-submitted preformatted data as a DataFrame
            try:
                snpsdf = pd.read_csv(
                    input_data,
                    index_col=0,
                    sep=None,
                    engine="python",
                    dtype=str,
                )
                # Need to clean up the dataframe if there is extra stuff in it
                # keep the first column, it's the index
                cols_to_keep = [snpsdf.columns[0]]
                for col in snpsdf.columns[1:]:
                    if col.startswith("rs"):
                        cols_to_keep.append(col)
                        snpsdf[col] = snpsdf[col].astype("object")
                return snpsdf[cols_to_keep]
            except:
                raise ValueError(
                    f"{input_data} is not a valid file or directory. Please provide a valid file or directory."
                )
    else:
        snpsdf = _input_to_dataframe(input_data, aisnpsdf)
    
    for col in snpsdf.columns:
        snpsdf[col] = snpsdf[col].astype("object")
    return snpsdf


def _input_to_dataframe(input_data, aisnpsdf):
    """Reads one file and returns a pandas DataFrame.

    :param aisnpsdf: A DataFrame of aisnps
    :type aisnpsdf: pandas DataFrame
    :param input_data: Path object to the file to be read or a SNPs DataFrame
    :type input_data: Path
    :return: A DataFrame of one record and many columns for each aisnp.
    :rtype: pandas DataFrame
    """
    # try to read a single file
    try:
        is_pathlike = bool(Path(input_data))
    except TypeError:
        is_pathlike = False
    if is_pathlike:
        try:
            snpsobj = SNPs(str(input_data))
            if snpsobj.count == 0:
                logger.debug(f"No snps found in the input_data, that could be OK if you have a csv")
                return None
            snpsdf = snpsobj.snps
        except FileNotFoundError:
            logger.critical(f"Could not find file {input_data}")
        sample_id = Path(input_data).name

    else:
        snpsdf = input_data
        sample_id = "sample"

    snpsdf = snpsdf.reset_index()
    snpsdf.rename(
        columns={"chrom": "chromosome", "pos": "position"},
        inplace=True,
    )
    # subset to aisnps
    snpsdf = aisnpsdf.merge(
        snpsdf, on=["rsid", "chromosome", "position"], how="left"
    )
    # inform user how many missing snps
    n_aisnps = snpsdf["genotype"].notnull().sum()
    n_aisnps_total = snpsdf.shape[0]
    logger.info(
        f"{sample_id} has a valid genotype for {n_aisnps} out of a possible {n_aisnps_total} ({(n_aisnps / n_aisnps_total) * 100}%)"
    )

    snpsdfT = pd.DataFrame(columns=snpsdf["rsid"].tolist())
    snpsdfT.loc[sample_id] = snpsdf["genotype"].tolist()

    return snpsdfT
