import ftplib
import sys
from pathlib import Path

from loguru import logger

from ezancestry.config import \
    thousand_genomes_directory as _thousands_genomes_directory


def download_thousand_genomes(thousand_genomes_directory=None):
    """Downloads the 1000 Genomes files to your local machine.

    :param thousand_genomes_directory: Full file path to a directory where you want to download the 1000 Genomes files.
    :type thousand_genomes_directory: str
    """

    if thousand_genomes_directory is None:
        thousand_genomes_directory = _thousands_genomes_directory
    thousand_genomes_directory = Path(thousand_genomes_directory)

    if not Path(thousand_genomes_directory).exists():
        sys.exit("Please enter a valid path: Exiting...")

    empty_directory = not any(Path(thousand_genomes_directory).iterdir())
    if empty_directory:
        if (
            input(
                "Are you sure you want to download 1000 Genomes data (13GB)? (y/n)"
            ).upper()
            != "Y"
        ):
            sys.exit("Exiting...")

    ftp_site = "ftp.1000genomes.ebi.ac.uk"
    filepath = "/vol1/ftp/release/20130502/supporting/bcf_files/"
    ftp = ftplib.FTP(ftp_site)
    ftp.login()
    ftp.cwd(filepath)

    logger.info("Downloading 1000 Genomes data, this will take about an hour")
    for chromosome in [str(_) for _ in range(1, 23)] + ["X", "Y"]:
        # bcf file
        bcf_file = f"ALL.chr{chromosome}.phase3_shapeit2_mvncall_integrated_v5.20130502.genotypes.bcf"
        if chromosome == "X" or chromosome == "Y":
            bcf_file = f"ALL.chr{chromosome}.phase3_shapeit2_mvncall_integrated.20130502.genotypes.bcf"

        # check if the file exists
        if not Path(thousand_genomes_directory).joinpath(bcf_file).exists():
            with open(
                Path(thousand_genomes_directory).joinpath(bcf_file), "wb"
            ) as fp:
                logger.info(f"Downloading chromosome {chromosome}...")
                ftp.retrbinary(f"RETR {bcf_file}", fp.write)
        else:
            logger.warning(
                f"{Path(thousand_genomes_directory).joinpath(bcf_file)} already exists, skipping..."
            )

        # index file
        index_file = f"ALL.chr{chromosome}.phase3_shapeit2_mvncall_integrated_v5.20130502.genotypes.bcf.csi"
        if chromosome == "X" or chromosome == "Y":
            index_file = f"ALL.chr{chromosome}.phase3_shapeit2_mvncall_integrated.20130502.genotypes.bcf.csi"

        # check if the file exists
        if not Path(thousand_genomes_directory).joinpath(index_file).exists():
            with open(
                Path(thousand_genomes_directory).joinpath(index_file), "wb"
            ) as fp:
                ftp.retrbinary(f"RETR {index_file}", fp.write)
        else:
            logger.warning(
                f"{Path(thousand_genomes_directory).joinpath(index_file)} already exists, skipping..."
            )
