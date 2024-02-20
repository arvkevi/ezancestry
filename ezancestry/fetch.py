import sys
from pathlib import Path

from loguru import logger
import pandas as pd
import pysam

from ezancestry.config import aisnps_directory as _aisnps_directory


# This should go from original 1000 Genomes vcf to aisnps df

def get_thousand_genomes_aisnps(aisnps_directory=None, aisnps_sets="kidd"):
    """Downloads the relevant 1000 Genomes genotypes.

    :param aisnps_directory: Full file path to a directory where you want to store the 1000 Genomes AISNPs data.
    :type aisnps_directory: str
    :param aisnps_sets: Which set of aisnp SNPs to extract. "kidd" or "seldin" or ["kidd", "seldin", "custom"], Defaults to "kidd".
    :type aisnps_sets: str or list
    """

    if aisnps_directory is None:
        aisnps_directory = _aisnps_directory
    aisnps_directory = Path(aisnps_directory)

    if not Path(aisnps_directory).exists():
        sys.exit("Please enter a valid path: Exiting...")

    if isinstance(aisnps_sets, str):
        # shortcut if aisnps is a string and it exists
        aisnps_1kg_filename = aisnps_directory.joinpath(f"{aisnps_sets}.1kG.csv")
        try:
            aisnps_1kg = pd.read_csv(aisnps_1kg_filename, dtype=str)
            logger.info(f"Loaded: {aisnps_sets}.1kG.csv")
            return aisnps_1kg
        except FileNotFoundError:
            pass
        aisnps_sets = [aisnps_sets]

    vcf_filepath = "https://ftp.ebi.ac.uk/1000g/ftp/release/20130502/"



    # Read the 1kg sample information from remote
    dfsamples = pd.read_csv(f"{vcf_filepath}integrated_call_samples_v3.20130502.ALL.panel", sep="\t")
    dfsamples.rename(columns={"pop": "population", "super_pop": "superpopulation"}, inplace=True)

    # Loop over the aisnps_sets
    for aisnps_set in aisnps_sets:
        aisnps_file = aisnps_directory.joinpath(f"{aisnps_set}.aisnp.txt")
        logger.info(f"Using: {aisnps_set}.aisnp.txt")

        # read the aisnp file
        try:
            aisnpsdf = pd.read_csv(aisnps_file, sep="\t", dtype=str)
        except FileNotFoundError:
            logger.error(
                f"Please check the path to the aisnps file ({aisnps_set}.aisnp.txt)"
            )
            sys.exit(1)

        # Create a DataFrame to store the 1000 Genomes data
        aisnps_1kg = pd.DataFrame(index=dfsamples["sample"])
        
        # Sort the integers and leave X to the end
        aisnpsdf.sort_values(by=["chromosome", "position"], inplace=True,
                     key=lambda x: x.map(lambda y: int(y) if y.isdigit() else 1000 + ord(y[0])))
        chrom_old = "0"
        for _, aim in aisnpsdf.iterrows():
            rsid = aim["rsid"]
            chrom = aim["chromosome"]
            pos = aim["position"]
            logger.info(f"Processing: {rsid} {chrom}:{pos}")

            # don't open the file again if we are still in the same chromosome
            if chrom != chrom_old:
                # Open the VCF file for the chromosome
                full_vcf_filepath = f"{vcf_filepath}ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
                variant_file = pysam.VariantFile(full_vcf_filepath, index_filename=full_vcf_filepath + ".tbi")
                chrom_old = chrom
            for variant in variant_file.fetch(chrom, int(pos)-1, int(pos)):
                if variant.pos == int(pos):
                    variant.ref = variant.ref
                    variant.alts = [alt for alt in variant.alts]
                    int_to_allele = {0: variant.ref}
                    int_to_allele.update({i+1: variant.alts[i] for i in range(len(variant.alts))})
                    genotypes = []
                    for sample in variant.samples:
                        genotypes.append(int_to_allele[variant.samples[sample]['GT'][0]] + int_to_allele[variant.samples[sample]['GT'][1]])
                    aisnps_1kg[rsid] = genotypes

        aisnps_1kg = aisnps_1kg.merge(dfsamples, left_index=True, right_on="sample", how="outer")
        cols_to_drop = aisnps_1kg.filter(like="Unnamed").columns.tolist()
        aisnps_1kg.drop(columns=cols_to_drop, inplace=True)
        aisnps_1kg.set_index("sample", inplace=True)
        aisnps_1kg.reset_index(inplace=True)
        
        if not aisnps_1kg_filename.exists():
            aisnps_1kg.to_csv(aisnps_1kg_filename, index=False)
            logger.info(f"Saved: {aisnps_set}.1kG.csv")
        else:
            logger.info(f"File already exists: {aisnps_set}.1kG.csv")
        for col in aisnps_1kg.columns:
            aisnps_1kg[col] = aisnps_1kg[col].astype("object")
        return aisnps_1kg

if __name__ == "__main__":
    get_thousand_genomes_aisnps()