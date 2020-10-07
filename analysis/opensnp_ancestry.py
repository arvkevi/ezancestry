""" Apply `ezancestry` to analyze ancestry of openSNP datadump files.

Outputs a CSV file with predicted super population and population probabilities
for each parsed file in the `openSNP <https://opensnp.org>`_ datadump.

Notes
-----
Relative paths assume script is being run from analysis dir.
"""

import random

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from snps import SNPs
from snps.resources import Resources
from snps.utils import Parallelizer, save_df_as_csv, create_dir

from app import (
    get_1kg_samples,
    encode_genotypes,
    dimensionality_reduction,
    filter_user_genotypes,
    impute_missing,
    vcf2df,
)

OUTPUT_DIR = "output"
AISNP_SET = (
    "Kidd et al. 55 AISNPs"
)  # {"Kidd et al. 55 AISNPs", "Seldin et al. 128 AISNPs"}
DIMENSIONALITY_REDUCTION_ALGORITHM = "PCA"  # {"PCA", "UMAP", "t-SNE"}

# create output directory for this example
create_dir(OUTPUT_DIR)

# assume `opensnp_datadump.current.zip` is found at this location
r = Resources(resources_dir="../data")


def main():
    # get filenames from openSNP data dump
    filenames = r.get_opensnp_datadump_filenames()

    # draw a sample from the observations
    random.seed(1)
    SAMPLE_SIZE = len(filenames)
    # SAMPLE_SIZE = 10
    samples = random.sample(range(len(filenames)), SAMPLE_SIZE)

    # get the 1000 genomes samples
    dfsamples = get_1kg_samples("../data/integrated_call_samples_v3.20130502.ALL.panel")

    aisnps_1kg = (
        vcf2df("../data/Kidd.55AISNP.1kG.vcf", dfsamples)
        if AISNP_SET == "Kidd et al. 55 AISNPs"
        else vcf2df("../data/Seldin.128AISNP.1kG.vcf", dfsamples)
    )

    # Encode 1kg data
    X_encoded, encoder = encode_genotypes(aisnps_1kg)

    # perform dimensionality reduction on the 1kg set
    X_reduced, reducer = dimensionality_reduction(
        X_encoded, algorithm=DIMENSIONALITY_REDUCTION_ALGORITHM
    )

    # predicted population
    knn_super_pop = KNeighborsClassifier(n_neighbors=9, weights="distance", n_jobs=1)
    knn_pop = KNeighborsClassifier(n_neighbors=9, weights="distance", n_jobs=1)

    # fit the knn before adding the user sample
    knn_super_pop.fit(X_reduced, dfsamples["super population"])
    knn_pop.fit(X_reduced, dfsamples["population"])

    # setup tasks for parallelizing / execution on multiple cores
    p = Parallelizer(parallelize=True)

    tasks = [
        {
            "file": filenames[i],
            "aisnps_1kg": aisnps_1kg,
            "X_encoded": X_encoded,
            "encoder": encoder,
            "reducer": reducer,
            "knn_super_pop": knn_super_pop,
            "knn_pop": knn_pop,
        }
        for i in samples
    ]

    # run tasks; results is a list of dicts
    results = p(process_file, tasks)

    # get rows for dataframe summarizing results
    rows = [row for row in results if row]

    df = pd.DataFrame(
        rows,
        columns=[
            "file",
            "source",
            "build",
            "build_detected",
            "chromosomes_summary",
            "snp_count",
            "AFR",
            "AMR",
            "EAS",
            "EUR",
            "SAS",
            "ACB",
            "ASW",
            "BEB",
            "CDX",
            "CEU",
            "CHB",
            "CHS",
            "CLM",
            "ESN",
            "FIN",
            "GBR",
            "GIH",
            "GWD",
            "IBS",
            "ITU",
            "JPT",
            "KHV",
            "LWK",
            "MSL",
            "MXL",
            "PEL",
            "PJL",
            "PUR",
            "STU",
            "TSI",
            "YRI",
        ],
    )

    save_df_as_csv(df, OUTPUT_DIR, "opensnp_ancestry.csv")


def process_file(task):
    file = task["file"]
    aisnps_1kg = task["aisnps_1kg"]
    X_encoded = task["X_encoded"]
    encoder = task["encoder"]
    reducer = task["reducer"]
    knn_super_pop = task["knn_super_pop"]
    knn_pop = task["knn_pop"]

    try:
        user_snps = SNPs(r.load_opensnp_datadump_file(file))

        # filter out files that likely don't have AISNPs
        if user_snps.count < 100000:
            return None

        d = {
            "file": file,
            "source": user_snps.source,
            "build": user_snps.build,
            "build_detected": user_snps.build_detected,
            "chromosomes_summary": user_snps.chromosomes_summary,
            "snp_count": user_snps.count,
        }

        # filter and encode the user record
        user_record, aisnps_1kg = filter_user_genotypes(user_snps.snps, aisnps_1kg)
        user_encoded = encoder.transform(user_record)
        X_encoded = np.concatenate((X_encoded, user_encoded))

        # impute the user record and reduce the dimensions
        user_imputed = impute_missing(X_encoded)
        user_reduced = reducer.transform([user_imputed])

        d.update(dict(get_predicted_probs(knn_super_pop, user_reduced).loc["user"]))
        d.update(dict(get_predicted_probs(knn_pop, user_reduced).loc["user"]))

        return d
    except:
        return None


def get_predicted_probs(knn, user_reduced):
    """ Get predicted ancestry probabilities for a user. """
    user_pop_probs = knn.predict_proba(user_reduced)
    return pd.DataFrame([user_pop_probs[0]], columns=knn.classes_, index=["user"])


if __name__ == "__main__":
    main()
