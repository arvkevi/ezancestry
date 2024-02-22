import webbrowser
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import typer
from loguru import logger
from sklearn.model_selection import train_test_split

from ezancestry import super_pop_codes, pop_codes
from ezancestry.config import aisnps_directory as _aisnps_directory
from ezancestry.config import aisnps_set as _aisnps_set
from ezancestry.config import models_directory as _models_directory
from ezancestry.config import population_level as _population_level
from ezancestry.fetch import get_thousand_genomes_aisnps
from ezancestry.evaluate import export_performance
from ezancestry.model import train as train_model, predict_ancestry
from ezancestry.process import process_user_input

import joblib


class PopulationLevel(str, Enum):
    population = "population"
    super_population = "superpopulation"


app = typer.Typer(
    help="The ezancestry command line tool makes it easy to quickly train models "
    "on 1000 genomes project data and view the performance of a "
    "ancestry-informative snp set (aisnps) at predicting genetic ancestry."
)

@app.command(short_help="Command to get the genetic variants from the 1000 genomes project into a DataFrame stored as a csv.")
def fetch(
    aisnps_directory: str = typer.Option(
        None,
        help="The path to the directory where the aisnps files are located.",
    ),
    aisnps_sets: Optional[str] = typer.Option(
        "kidd"
        "The name of the aisnp set to use. To start, choose either "
        "'kidd' or 'seldin'. The default value in conf.ini is 'kidd'."
    ),
):
    """
    This command will download the 1000 genomes project data and save it as a csv file in the aisnps directory.
    If you have a file of AISNPs named custom.aisnp.txt in the aisnps directory, you can run the following command to download the 1000 genomes project data for your custom AISNPs:

    $ ezancestry fetch --aisnps-sets custom
    """
    if aisnps_directory is None:
        aisnps_directory = _aisnps_directory
    aisnps_directory = Path(aisnps_directory)

    try:
        get_thousand_genomes_aisnps(aisnps_directory=aisnps_directory, aisnps_sets=aisnps_sets)
    except Exception as e:
        logger.error(e)
        return

    return


@app.command(short_help="Build and evaluate a new model from a set of aisnps.")
def train(
    models_directory: str = typer.Option(
        None, help="The path to the directory to save the model to."
    ),
    aisnps_directory: str = typer.Option(
        None,
        help="The path to the directory where the aisnps files are located.",
    ),
    population_level: PopulationLevel = typer.Option(
        PopulationLevel.super_population,
        help="The granularity of genetic ancestry you want to predict.",
    ),
    aisnps_set: Optional[str] = typer.Option(
        None,
        help="The name of the aisnp set to use. To start, choose either "
        "'kidd' or 'seldin'. The default value in conf.ini is 'kidd'."
        "\n*If using your aisnp set, this value will be the in the naming"
        "convention for all the new model files that are created*",
    ),
):
    """
    This will create a model for your custom aisnps set using the default pipeline.
    If you create a custom aisnp file here: ~/.ezancestry/data/aisnps/custom.aisnp.txt
    and then run the train command, you will build a model from the snps in that file:

    $ ezancestry train --aisnps-set custom

    See github.com/ezancestry/ezancestry/data/aisnps/custom.aisnp.txt for an example of a custom aisnp file.
    
    * Default arguments are from the ~/.ezancestry/conf.ini file. *
    """

    if models_directory is None:
        models_directory = _models_directory
    if aisnps_directory is None:
        aisnps_directory = _aisnps_directory
    if population_level is None:
        population_level = _population_level
    if aisnps_set is None:
        aisnps_set = _aisnps_set

    models_directory = Path(models_directory)
    aisnps_directory = Path(aisnps_directory)

    # download 1kg
    dfsnps = get_thousand_genomes_aisnps(aisnps_directory=aisnps_directory, aisnps_sets=aisnps_set)

    # process data
    labels = dfsnps[population_level]
    dfsnps.drop(
        columns=["population", "superpopulation", "gender"], inplace=True
    )

    # split training and test data
    train_df, test_df, y_train, y_test = train_test_split(
        dfsnps,
        labels,
        test_size=0.4,
        stratify=labels,
        random_state=42,
    )

    model = train_model(
        train_df,
        y_train,
        sklearn_pipeline=None,
        models_directory=models_directory,
        aisnps_set=aisnps_set,
        population_level=population_level,
        overwrite_model=True,
    )

    cv_report, holdout_report = export_performance(
        train_df,
        test_df,
        y_train,
        y_test,
        model=model,
        models_directory=models_directory,
        aisnps_directory=aisnps_directory,
        population_level=population_level,
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
        help="The path to the directory where the aisnps files are located.",
    ),
    aisnps_set: Optional[str] = typer.Option(
        None,
        help="The name of the aisnp set to use. To start, choose either "
        "'kidd' or 'seldin'. The default value in conf.ini is 'kidd'."
        "\n*If using your aisnp set, this value will be the in the naming "
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
    if aisnps_set is None:
        aisnps_set = _aisnps_set
    if output_directory is None:
        output_directory = Path.cwd()
    
    output_directory = Path(output_directory)
    models_directory = Path(_models_directory)
    aisnps_directory = Path(aisnps_directory)
    overall_predictions = pd.DataFrame()
    for population_level in ["population", "superpopulation"]:
        # Load an aisnps_set model from the default models directory
        model_path = models_directory.joinpath(
            f"{aisnps_set}_{population_level}.pkl"
        )

        snpsdf = process_user_input(input_data, aisnps_directory, aisnps_set)
        index = snpsdf.index
        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            logger.error(
                f"Could not find the model at {model_path}. Please train a model first."
            )
        snpsdf = snpsdf[model.feature_names_in_]

        predictions = predict_ancestry(snpsdf, model)
        predictions.rename(columns={"predicted_ancestry": f"predicted_ancestry_{population_level}"}, inplace=True)
        predictions.index = index
        if population_level == "superpopulation":
            # just get the predictions
            predictions = predictions[model.classes_.tolist() + ["predicted_ancestry_superpopulation"]]
        overall_predictions = pd.concat([overall_predictions, predictions], axis=1)


    try:
        if "PCA" or "pca" in model.named_steps:
            for i, step in enumerate(model.named_steps):
                if "PCA" in step or "pca" in step:
                    break
            pca = model[:i+1].transform(snpsdf)
            overall_predictions["component1"] = pca[:, 0]
            overall_predictions["component2"] = pca[:, 1]
            overall_predictions["component3"] = pca[:, 2]
    except Exception as e:
        logger.debug(e)
        logger.warning(
            "The model does not have PCA as a step, so no PCA components were added to the predictions. Or there were not enough PC components to add to the predictions."
        )

    if write_predictions:
        overall_predictions.to_csv(output_directory.joinpath("predictions.csv"))
        # store the settings in a commented header
        line = (
            f"#{aisnps_set},{model_path},{aisnps_directory}"
        )
        with open(output_directory.joinpath("predictions.csv"), "r+") as f:
            content = f.read()
            f.seek(0, 0)
            f.write(line.rstrip("\r\n") + "\n" + content)

        logger.info(
            f"Predictions written to {output_directory}/predictions.csv"
        )

    return overall_predictions


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
                (aisnps_set, models_directory, aisnps_directory,) = (
                    line.strip("#").strip("\n").split(",")
                )

    predictions = pd.read_csv(predictions_file, index_col=0, comment="#")
    columns = [
        "component1",
        "component2",
        "component3",
        "predicted_ancestry_population",
        "predicted_ancestry_superpopulation",
    ]
    predictions = predictions[columns]
    aisnps_file = Path(aisnps_directory).joinpath(
        f"{aisnps_set}.1kG.csv"
    )
    # don't save the results of these predictions
    aisnps_results = predict(
        aisnps_file,
        output_directory=None,
        models_directory=models_directory,
        aisnps_directory=aisnps_directory,
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
        population_level_column = "predicted_ancestry_superpopulation"
    else:
        population_level_column = "predicted_ancestry_population"

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

