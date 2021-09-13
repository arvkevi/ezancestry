import webbrowser
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import typer
from loguru import logger
from sklearn.model_selection import train_test_split

from ezancestry.aisnps import extract_aisnps
from ezancestry.config import aisnps_directory as _aisnps_directory
from ezancestry.config import aisnps_set as _aisnps_set
from ezancestry.config import algorithm as _algorithm
from ezancestry.config import k as _k
from ezancestry.config import models_directory as _models_directory
from ezancestry.config import n_components as _n_components
from ezancestry.config import population_level as _population_level
from ezancestry.config import samples_directory as _samples_directory
from ezancestry.config import \
    thousand_genomes_directory as _thousand_genomes_directory
from ezancestry.dimred import dimensionality_reduction
from ezancestry.evaluate import export_performance
from ezancestry.fetch import download_thousand_genomes
from ezancestry.model import predict_ancestry, train
from ezancestry.process import (encode_genotypes, get_1kg_labels,
                                process_user_input, vcf2df)


class PopulationLevel(str, Enum):
    population = "population"
    super_population = "superpopulation"


app = typer.Typer(
    help="The ezancestry command line tool makes it easy to quickly train models "
    "on 1000 genomes project data and view the performance of a "
    "ancestry-informative snp set (AISNPs) at predicting genetic ancestry."
)


@app.command(short_help="Build and evaluate a new model from a set of AISNPs.")
def build_model(
    models_directory: str = typer.Option(
        None, help="The path to the directory to save the model to."
    ),
    aisnps_directory: str = typer.Option(
        None,
        help="The path to the directory where the AISNPs files are located.",
    ),
    n_components: int = typer.Option(
        None,
        help="The number of components to use in the PCA dimensionality reduction.",
    ),
    k: int = typer.Option(
        None, help="The number of nearest neighbors to use in the KNN model."
    ),
    thousand_genomes_directory: str = typer.Option(
        None, help="The path to the 1000 genomes directory."
    ),
    samples_directory: str = typer.Option(
        None, help="The path to the directory containing the samples."
    ),
    population_level: PopulationLevel = typer.Option(
        PopulationLevel.super_population,
        help="The granularity of genetic ancestry you want to predict.",
    ),
    algorithm: str = typer.Option(
        None,
        help="The dimensionality reduction algorithm to use. Use one of pca|umap|nca",
    ),
    aisnps_set: Optional[str] = typer.Option(
        None,
        help="The name of the AISNP set to use. To start, choose either "
        "'Kidd' or 'Seldin'. The default value in conf.ini is 'Kidd'."
        "\n*If using your AISNP set, this value will be the in the naming"
        "convention for all the new model files that are created*",
    ),
):
    """
    For example, if you create a custom AISNP file here: ~/.ezancestry/data/aisnps/custom.AISNP.txt
    and then run the build-model command, you will build a model from the snps in that file:

    $ ezancestry build-model --aisnps-set custom

    See github.com/ezancestry/ezancestry/data/aisnps/custom.AISNP.txt for an example of a custom AISNP file.

    * Note that the 1000 genomes dataset is required for this function to work. *

    * Default arguments are from the ~/.ezancestry/conf.ini file. *
    """

    if models_directory is None:
        models_directory = _models_directory
    if aisnps_directory is None:
        aisnps_directory = _aisnps_directory
    if population_level is None:
        population_level = _population_level
    if algorithm is None:
        algorithm = _algorithm
    if n_components is None:
        n_components = _n_components
    if k is None:
        k = _k
    if thousand_genomes_directory is None:
        thousand_genomes_directory = _thousand_genomes_directory
    if aisnps_set is None:
        aisnps_set = _aisnps_set
    if samples_directory is None:
        samples_directory = _samples_directory

    models_directory = Path(models_directory)
    aisnps_directory = Path(aisnps_directory)
    samples_directory = Path(samples_directory)
    thousand_genomes_directory = Path(thousand_genomes_directory)

    # download 1kG
    download_thousand_genomes(thousand_genomes_directory)
    # extract snps
    aisnps_file = Path(aisnps_directory).joinpath(f"{aisnps_set}.AISNP.txt")
    extract_aisnps(thousand_genomes_directory, aisnps_file, aisnps_set)

    # process data
    dfsamples = get_1kg_labels(samples_directory)
    vcf_fname = Path(aisnps_directory).joinpath(f"{aisnps_set}.AISNP.1kG.vcf")
    dfsnps = vcf2df(vcf_fname, dfsamples)
    labels = dfsnps[population_level]
    dfsnps.drop(
        columns=["population", "superpopulation", "gender"], inplace=True
    )

    # split the training and test data
    train_df, test_df, y_train, y_test = train_test_split(
        dfsnps,
        labels,
        test_size=0.4,
        stratify=labels,
        random_state=42,
    )

    # fit & write models on 1kG
    dfencoded_train = encode_genotypes(
        train_df, aisnps_set=aisnps_set, overwrite_encoder=True
    )
    dfencoded_test = encode_genotypes(
        test_df, aisnps_set=aisnps_set, overwrite_encoder=False
    )

    dfreduced_train = dimensionality_reduction(
        dfencoded_train,
        algorithm=algorithm,
        aisnps_set=aisnps_set,
        overwrite_model=True,
        labels=y_train,
        population_level=population_level,
    )
    dfreduced_test = dimensionality_reduction(
        dfencoded_test,
        algorithm=algorithm,
        aisnps_set=aisnps_set,
        overwrite_model=False,
        labels=y_train,
        population_level=population_level,
    )

    knn_model = train(
        dfreduced_train,
        y_train,
        algorithm=algorithm,
        aisnps_set=aisnps_set,
        k=k,
        population_level=population_level,
        overwrite_model=True,
    )

    cv_report, holdout_report = export_performance(
        dfreduced_train,
        dfreduced_test,
        y_train,
        y_test,
        models_directory=models_directory,
        aisnps_directory=aisnps_directory,
        population_level=population_level,
        algorithm=algorithm,
        aisnps_set=aisnps_set,
    )

    return cv_report, holdout_report


@app.command(
    short_help="Predict your genetic ancestry or predict it for a cohort of samples."
)
def predict(
    input_data: str = typer.Argument(
        ...,
        help="Can be a file path to raw genetic data (23andMe, ancestry.com, .vcf) file, a path to a directory containing several raw genetic files, or a (tab or comma) delimited file with sample ids as rows and snps as columns.",
    ),
    output_directory: str = typer.Option(
        None, help="The directory where to write the prediction results file"
    ),
    write_predictions: bool = typer.Option(
        True,
        help="If True, write the predictions to a file. If False, return the predictions as a dataframe.",
    ),
    models_directory: str = typer.Option(
        None,
        help="The path to the directory where the model files are located.",
    ),
    aisnps_directory: str = typer.Option(
        None,
        help="The path to the directory where the AISNPs files are located.",
    ),
    n_components: int = typer.Option(
        None,
        help="The number of components to use in the PCA dimensionality reduction.",
    ),
    k: int = typer.Option(
        None, help="The number of nearest neighbors to use in the KNN model."
    ),
    thousand_genomes_directory: str = typer.Option(
        None, help="The path to the 1000 genomes directory."
    ),
    samples_directory: str = typer.Option(
        None, help="The path to the directory containing the samples."
    ),
    algorithm: str = typer.Option(
        None,
        help="The dimensionality reduction algorithm to use. Choose pca|umap|nca",
    ),
    aisnps_set: Optional[str] = typer.Option(
        None,
        help="The name of the AISNP set to use. To start, choose either "
        "'Kidd' or 'Seldin'. The default value in conf.ini is 'Kidd'."
        "\n*If using your AISNP set, this value will be the in the naming "
        "convention for all the new model files that are created*",
    ),
):
    """
    Predict ancestry from genetic data.

    * Default arguments are from the ~/.ezancestry/conf.ini file. *
    """

    if models_directory is None:
        models_directory = _models_directory
    if aisnps_directory is None:
        aisnps_directory = _aisnps_directory
    if algorithm is None:
        algorithm = _algorithm
    if n_components is None:
        n_components = _n_components
    if k is None:
        k = _k
    if thousand_genomes_directory is None:
        thousand_genomes_directory = _thousand_genomes_directory
    if aisnps_set is None:
        aisnps_set = _aisnps_set
    if samples_directory is None:
        samples_directory = _samples_directory
    if output_directory is None:
        output_directory = Path.cwd()

    output_directory = Path(output_directory)
    models_directory = Path(models_directory)
    aisnps_directory = Path(aisnps_directory)
    samples_directory = Path(samples_directory)
    thousand_genomes_directory = Path(thousand_genomes_directory)

    snpsdf = process_user_input(input_data, aisnps_directory, aisnps_set)
    index = snpsdf.index
    snpsdf = encode_genotypes(
        snpsdf,
        aisnps_set=aisnps_set,
        models_directory=models_directory,
        aisnps_directory=aisnps_directory,
        overwrite_encoder=False,
    )
    pop_predictions = pd.DataFrame()
    superpop_predictions = pd.DataFrame()
    for pop_level in PopulationLevel:
        pop_level = pop_level
        dimreddf = dimensionality_reduction(
            snpsdf,
            algorithm=algorithm,
            aisnps_set=aisnps_set,
            population_level=pop_level,
            models_directory=models_directory,
            overwrite_model=False,
        )
        knn = models_directory.joinpath(
            f"knn.{algorithm}.{aisnps_set}.{pop_level}.bin"
        )
        pop_level_predictions = predict_ancestry(dimreddf, knn)
        if pop_level == "population":
            pop_predictions = pop_predictions.append(pop_level_predictions)
        else:
            superpop_predictions = superpop_predictions.append(
                pop_level_predictions
            )
    predictions = pop_predictions.merge(
        superpop_predictions.drop(columns=["x", "y", "z"]),
        left_index=True,
        right_index=True,
        suffixes=("_population", "_superpopulation"),
    )

    predictions.rename(
        columns={"x": "component1", "y": "component2", "z": "component3"},
        inplace=True,
    )
    predictions.set_index(index, inplace=True)

    if write_predictions:
        predictions.to_csv(output_directory.joinpath("predictions.csv"))
        # store the settings in a commented header
        line = (
            f"#{algorithm},{aisnps_set},{models_directory},{aisnps_directory}"
        )
        with open(output_directory.joinpath("predictions.csv"), "r+") as f:
            content = f.read()
            f.seek(0, 0)
            f.write(line.rstrip("\r\n") + "\n" + content)

        logger.info(
            f"Predictions written to {output_directory}/predictions.csv"
        )

    return predictions


@app.command(
    short_help="Plot the output of the `predict` command against the 1000 genomes samples in 3D"
)
def plot(
    predictions_file: str = typer.Argument(
        ...,
        help="The predictions.csv file from the output of ezancestry predict command (or using predict when using ezancestry as a library).",
    ),
    population_level: PopulationLevel = typer.Option(
        None,
        help="The population level to plot. Can be either 'population' or 'superpopulation'.",
    ),
    output_directory: str = typer.Option(
        None, help="The directory where to write the html plot file"
    ),
):
    """Reads the output of the ezancestry predict command"""

    if output_directory is None:
        output_directory = Path.cwd()
    output_directory = Path(output_directory)
    if population_level is None:
        population_level = _population_level

    with open(predictions_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                (algorithm, aisnps_set, models_directory, aisnps_directory,) = (
                    line.strip("#").strip("\n").split(",")
                )

    predictions = pd.read_csv(predictions_file, index_col=0, comment="#")
    columns = [
        "component1",
        "component2",
        "component3",
        "predicted_population_population",
        "predicted_population_superpopulation",
    ]
    predictions = predictions[columns]
    aisnps_file = Path(aisnps_directory).joinpath(
        f"thousand_genomes.{aisnps_set}.dataframe.csv"
    )
    # don't save the results of these predictions
    aisnps_results = predict(
        aisnps_file,
        output_directory=None,
        models_directory=models_directory,
        aisnps_directory=aisnps_directory,
        thousand_genomes_directory=None,
        samples_directory=None,
        algorithm=algorithm,
        aisnps_set=aisnps_set,
        write_predictions=False,
    )
    aisnps_results = aisnps_results[columns]

    # add a label column to each dataframe that indicates it was user uploaded or not
    aisnps_results["label"] = "thousand_genomes_sample"
    predictions["label"] = "user_uploaded_sample"

    # merge the two dataframes
    df = pd.concat([predictions, aisnps_results])
    df.reset_index(inplace=True)
    df.rename(columns={"index": "sample"}, inplace=True)

    # plot the data
    color_discrete_map = {"user_uploaded_sample": "rgb(0,0,0)"}
    df["size"] = 16
    df.loc[df["label"] == "user_uploaded_sample", "size"] = 150

    if population_level == "superpopulation":
        population_level_column = "predicted_population_superpopulation"
    else:
        population_level_column = "predicted_population_population"

    fig = px.scatter_3d(
        df,
        x="component1",
        y="component2",
        z="component3",
        hover_name="sample",
        color=population_level_column,
        color_discrete_map=color_discrete_map,
        symbol=population_level_column,
        height=800,
        size="size",
        opacity=1.0,
        color_discrete_sequence=[
            "#008fd5",
            "#fc4f30",
            "#e5ae38",
            "#6d904f",
            "#810f7c",
        ],
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=18),
        )
    )

    fig.write_html(output_directory.joinpath("plot.html"))

    webbrowser.open(f"file://{output_directory.joinpath('plot.html')}")

    return fig


@app.command(
    short_help="Generate the data and populate the models in the data/* directories"
)
def generate_dependencies(
    models_directory: str = typer.Option(
        None,
        help="The path to the directory where the model files are located.",
    ),
    aisnps_directory: str = typer.Option(
        None,
        help="The path to the directory where the AISNPs files are located.",
    ),
    thousand_genomes_directory: str = typer.Option(
        None, help="The path to the 1000 genomes directory."
    ),
    samples_directory: str = typer.Option(
        None, help="The path to the directory containing the samples."
    ),
):
    """
    Generate the data and populate the models in the data/* directories.
    """
    if models_directory is None:
        models_directory = _models_directory
    if aisnps_directory is None:
        aisnps_directory = _aisnps_directory
    if thousand_genomes_directory is None:
        thousand_genomes_directory = _thousand_genomes_directory
    if samples_directory is None:
        samples_directory = _samples_directory

    models_directory = Path(models_directory)
    aisnps_directory = Path(aisnps_directory)
    samples_directory = Path(samples_directory)
    thousand_genomes_directory = Path(thousand_genomes_directory)

    k = _k
    n_components = _n_components
    algorithms = ["PCA", "UMAP", "NCA"]
    population_levels = ["population", "superpopulation"]
    aisnps_sets = ["Kidd", "Seldin"]

    for algorithm in algorithms:
        for population_level in population_levels:
            for aisnps_set in aisnps_sets:
                _, _ = build_model(
                    models_directory,
                    aisnps_directory,
                    n_components,
                    k,
                    thousand_genomes_directory,
                    samples_directory,
                    population_level,
                    algorithm,
                    aisnps_set,
                )
