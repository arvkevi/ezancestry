import configparser
import importlib.resources as pkg_resources
import os
import shutil
from pathlib import Path

from loguru import logger

from ezancestry import __project__

# setup the project directory in the user's home diretory
project_directory = Path.home().joinpath(f".{__project__}")
Path(project_directory).mkdir(parents=True, exist_ok=True)

default_data_directory = Path(project_directory).joinpath("data")
default_models_directory = default_data_directory.joinpath("models")
default_aisnps_directory = default_data_directory.joinpath("aisnps")
default_samples_directory = default_data_directory.joinpath("samples")
default_thousand_genomes_directory = default_data_directory.joinpath(
    "thousand_genomes"
)

default_data_directory.mkdir(parents=True, exist_ok=True)
default_models_directory.mkdir(parents=True, exist_ok=True)
default_aisnps_directory.mkdir(parents=True, exist_ok=True)
default_samples_directory.mkdir(parents=True, exist_ok=True)
default_thousand_genomes_directory.mkdir(parents=True, exist_ok=True)

full_config_file_path = Path(project_directory).joinpath("conf.ini")

config = configparser.ConfigParser()

# make sure the non-1kG default directories have data in them
if not any(default_models_directory.iterdir()):
    # pkg_resources copy data from the package to the user's home directory
    logger.info(
        f"Copying models to {default_models_directory}, this only happens once..."
    )
    for fileobj in pkg_resources.contents("data.models"):
        if not str(fileobj).endswith(".bin"):
            continue
        with pkg_resources.path("data.models", fileobj) as file_to_copy:
            shutil.copy(file_to_copy, default_models_directory)


if not any(default_aisnps_directory.iterdir()):
    # pkg_resources copy data from the package to the user's home directory
    logger.info(
        f"Copying aisnps data to {default_aisnps_directory}, this only happens once..."
    )
    for fileobj in pkg_resources.contents("data.aisnps"):
        if fileobj == "__init__.py" or fileobj == "__pycache__":
            continue
        with pkg_resources.path("data.aisnps", fileobj) as file_to_copy:
            shutil.copy(file_to_copy, default_aisnps_directory)

if not any(default_samples_directory.iterdir()):
    # pkg_resources copy data from the package to the user's home directory
    logger.info(
        f"Copying samples data to {default_samples_directory}, this only happens once..."
    )
    for fileobj in pkg_resources.contents("data.samples"):
        if fileobj == "__init__.py" or fileobj == "__pycache__":
            continue
        with pkg_resources.path("data.samples", fileobj) as file_to_copy:
            shutil.copy(file_to_copy, default_samples_directory)

# Set default values
config["directories"] = {
    "data_directory": default_data_directory,
    "models_directory": default_models_directory,
    "aisnps_directory": default_aisnps_directory,
    "samples_directory": default_samples_directory,
    "thousand_genomes_directory": default_thousand_genomes_directory,
}
config["general"] = {
    "population_level": "superpopulation",
    "aisnps_set": "Kidd",
}
config["dimensionality_reduction"] = {
    "algorithm": "PCA",
    "n_components": 3,
}
config["nearest_neighbors"] = {
    "k": 9,
}

if (
    not os.path.exists(full_config_file_path)
    or os.stat(full_config_file_path).st_size == 0
):
    with open(full_config_file_path, "w") as configfile:
        config.write(configfile)

# Read config file
config.read(full_config_file_path)
data_directory = config["directories"]["data_directory"]
models_directory = config["directories"]["models_directory"]
aisnps_directory = config["directories"]["aisnps_directory"]
samples_directory = config["directories"]["samples_directory"]
thousand_genomes_directory = config["directories"]["thousand_genomes_directory"]
population_level = config["general"]["population_level"]
aisnps_set = config["general"]["aisnps_set"]
algorithm = config["dimensionality_reduction"]["algorithm"]
n_components = int(config["dimensionality_reduction"]["n_components"])
k = int(config["nearest_neighbors"]["k"])

if not data_directory:
    data_directory = Path(__file__).parents[1].joinpath("data")
if not models_directory:
    models_directory = data_directory.joinpath("models")
if not aisnps_directory:
    aisnps_directory = data_directory.joinpath("aisnps")
if not samples_directory:
    samples_directory = data_directory.joinpath("samples")
if not thousand_genomes_directory:
    thousand_genomes_directory = data_directory.joinpath("thousand_genomes")
