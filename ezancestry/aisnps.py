import sys
from pathlib import Path

import pandas as pd
from cyvcf2 import VCF, Writer
from loguru import logger

from ezancestry.config import aisnps_directory as _aisnps_directory
from ezancestry.config import aisnps_set as _aisnps_set
from ezancestry.config import data_directory as _data_directory
from ezancestry.config import \
    thousand_genomes_directory as _thousand_genomes_directory
from ezancestry.fetch import download_thousand_genomes


def extract_aisnps(
    thousand_genomes_directory=None,
    aisnps_file=None,
    aisnps_set=None,
    aisnps_directory=None,
):
    """Extract the AISNP SNPs from the 1000 Genomes data. The thosuand_genomes_directory must be populated with data.

    :param thousand_genomes_directory: Full path to the directory where the 1000 genomes bcf files are located.
    :type thousand_genomes_directory: str
    :param aisnps_file: Full path to the file containing the AISNP SNPs.
    :type aisnps_file: str
    :param aisnps_set: Which set of AISNP SNPs to extract.
    :type aisnps_set: str
    :param aisnps_directory: Full path to the directory where the AISNP SNPs will be written.
    :type aisnps_directory: str
    """

    # default to the config if None
    if thousand_genomes_directory is None:
        thousand_genomes_directory = _thousand_genomes_directory
    thousand_genomes_directory = Path(thousand_genomes_directory)

    if aisnps_directory is None:
        aisnps_directory = _aisnps_directory
    aisnps_directory = Path(aisnps_directory)

    if aisnps_file is None:
        aisnps_file = aisnps_directory.joinpath(f"{aisnps_set}.AISNP.txt")
        logger.info(f"Using: {aisnps_set}.AISNP.txt")
    if aisnps_set is None:
        aisnps_set = _aisnps_set

    # read the AISNP file
    try:
        df = pd.read_csv(aisnps_file, sep="\t", dtype=str)
    except FileNotFoundError:
        logger.error(
            "Please check the path to the AISNPs file (KIDD.AISNP.txt)"
        )
        sys.exit(1)

    logger.info(
        f"Looking for 1000 genomes data in: {thousand_genomes_directory}"
    )
    bcf_fname = f"ALL.chr{1}.phase3_shapeit2_mvncall_integrated_v5.20130502.genotypes.bcf"
    bcf_file = thousand_genomes_directory.joinpath(bcf_fname)
    bcf = VCF(bcf_file)
    outfile = aisnps_directory.joinpath(f"{aisnps_set}.AISNP.1kG.vcf")
    w = Writer(outfile, bcf)
    for _, aim in df.iterrows():
        rsid = aim["rsid"]
        chrom = aim["chromosome"]
        pos = aim["position_hg19"]
        bcf_fname = f"ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5.20130502.genotypes.bcf"
        bcf_file = thousand_genomes_directory.joinpath(bcf_fname)
        bcf = VCF(bcf_file)
        snp_writer = 0
        for variant in bcf("{}:{}-{}".format(chrom, pos, pos)):
            if variant.POS == int(pos):
                w.write_record(variant)
                snp_writer += 1
        if snp_writer == 0:
            logger.warning(
                f"Couldn't find {rsid} on chromosome {chrom} position {pos}"
            )
    w.close()
    bcf.close()
    logger.info(f"Successfully wrote {aisnps_set}.AISNP.1kG.vcf")


def _generate_aisnps(
    thousand_genomes_directory=None, aisnps_set=None, aisnps_directory=None
):
    """A utility function to download the 1000 Genomes data, if necessary. Then create the AISNPs vcf files.

    :param thousand_genomes_directory: Full path to the directory where the 1000 genomes bcf files are located.
    :type thousand_genomes_directory: str
    :param aisnps_set: Which set of AISNP SNPs to extract.
    :type aisnps_set: str
    :param aisnps_directory: Full path to the directory where the AISNP SNPs will be written.
    :type aisnps_directory: str
    """

    # default to the config if None
    if thousand_genomes_directory is None:
        thousand_genomes_directory = _thousand_genomes_directory
    if aisnps_directory is None:
        aisnps_directory = _aisnps_directory
    if aisnps_set is None:
        aisnps_set = _aisnps_set

    download_thousand_genomes(thousand_genomes_directory)
    extract_aisnps(
        thousand_genomes_directory,
        aisnps_directory=aisnps_directory,
        aisnps_set="Kidd",
    )
    extract_aisnps(
        thousand_genomes_directory,
        aisnps_directory=aisnps_directory,
        aisnps_set="Seldin",
    )
