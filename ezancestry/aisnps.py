import sys
from pathlib import Path

import pandas as pd
from cyvcf2 import VCF, Writer
from loguru import logger

from ezancestry.fetch import download_thousand_genomes
from ezancestry.settings import aisnps_directory

# Add additional aisnps here


def extract_kidd_aisnps(thousand_genomes_directory):
    """Extracts ancestry-informative snps from the 1000 genomes bcf files.
    Writes a .vcf file of 1000 Genomes snps from only the AISNP locations
    defined by Kidd et al.

    :param thousand_genomes_directory: Full path to the directory where the 1000 genomes bcf files are located.
    :type thousand_genomes_directory: str
    """
    # read the AISNP genomic locations defined by Kidd et al
    kidd_aisnps_file = aisnps_directory.joinpath("Kidd_55_AISNPs.txt")
    try:
        df55 = pd.read_csv(kidd_aisnps_file, sep="\t")
    except FileNotFoundError:
        logger.error(
            "Please pass the full path to the Kidd AISNPs file (Kidd_55_AISNPs.txt)"
        )
        sys.exit(1)

    bcf_fname = "ALL.chr{}.phase3_shapeit2_mvncall_integrated_v5.20130502.genotypes.bcf".format(
        1
    )
    bcf_file = Path(thousand_genomes_directory).joinpath(bcf_fname)
    bcf = VCF(bcf_file)
    outfile = aisnps_directory.joinpath("Kidd.55AISNP.1kG.vcf")
    w = Writer(outfile, bcf)
    for _, aim in df55.iterrows():
        chrom = str(aim["Chr"])
        pos = str(aim["Build 37 nt position"])
        pos = pos.replace(",", "")
        rsid = str(aim["dbSNP rs#"])
        bcf_fname = "ALL.chr{}.phase3_shapeit2_mvncall_integrated_v5.20130502.genotypes.bcf".format(
            chrom
        )
        bcf_file = Path(thousand_genomes_directory).joinpath(bcf_fname)
        bcf = VCF(bcf_file)
        for variant in bcf("{}:{}-{}".format(chrom, pos, pos)):
            if variant.POS == int(pos):
                w.write_record(variant)
    w.close()
    bcf.close()
    logger.info("Successfully wrote Kidd.55AISNP.1kG.vcf")


def extract_seldin_aisnps(thousand_genomes_directory):
    """Extracts ancestry-informative snps from the 1000 genomes bcf files.
    Writes a .vcf file of 1000 Genomes snps from only the AISNP locations
    defined by Seldin et al.

    :param thousand_genomes_directory: Full path to the directory where the 1000 genomes bcf files are located.
    :type data_directory: str
    """
    # read the AISNP genomic locations defined by Kidd et al
    seldin_aisnps_file = aisnps_directory.joinpath(
        "report_Seldin_128_AISNPs.grch36.txt.xls"
    )
    try:
        df128 = pd.read_csv(seldin_aisnps_file, sep="\t")
    except FileNotFoundError:
        logger.error(
            "Please pass the full path to the Kidd AISNPs file (report_Seldin_128_AISNPs.grch36.txt.xls')"
        )
        sys.exit(1)

    bcf_fname = f"ALL.chr{1}.phase3_shapeit2_mvncall_integrated_v5.20130502.genotypes.bcf"
    bcf_file = Path(thousand_genomes_directory).joinpath(bcf_fname)
    bcf = VCF(bcf_file)
    outfile = aisnps_directory.joinpath("Seldin.128AISNP.1kG.vcf")
    w = Writer(outfile, bcf)
    for _, aim in df128.iterrows():
        chrom = aim["source_id"]
        pos = aim["mapped_start"]
        bcf_fname = f"ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5.20130502.genotypes.bcf"

        bcf_file = Path(thousand_genomes_directory).joinpath(bcf_fname)
        bcf = VCF(bcf_file)
        for variant in bcf(f"{chrom}:{pos}-{pos}"):
            if variant.POS == int(pos):
                w.write_record(variant)
    w.close()
    bcf.close()
    logger.info("Successfully wrote Seldin.128AISNP.1kG.vcf")


def generate_aisnps(thousand_genomes_directory):
    """Download the 1000 Genomes data, if necessary. Then create the AISNPs vcf files.

    :param bcf_directory: Full path to the directory where the 1000 genomes bcf files are located.
    :type data_directory: str
    """
    download_thousand_genomes(thousand_genomes_directory)
    extract_kidd_aisnps(thousand_genomes_directory)
    extract_seldin_aisnps(thousand_genomes_directory)
