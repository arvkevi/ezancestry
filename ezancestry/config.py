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

default_data_directory.mkdir(parents=True, exist_ok=True)
default_models_directory.mkdir(parents=True, exist_ok=True)
default_aisnps_directory.mkdir(parents=True, exist_ok=True)

full_config_file_path = Path(project_directory).joinpath("conf.ini")

config = configparser.ConfigParser()

# make sure the non-1kg default directories have data in them
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

# Set default values
config["directories"] = {
    "data_directory": default_data_directory,
    "models_directory": default_models_directory,
    "aisnps_directory": default_aisnps_directory,
}
config["general"] = {
    "population_level": "superpopulation",
    "aisnps_set": "kidd",
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
population_level = config["general"]["population_level"]
aisnps_set = config["general"]["aisnps_set"]

if not data_directory:
    data_directory = Path(__file__).parents[1].joinpath("data")
if not models_directory:
    models_directory = data_directory.joinpath("models")
if not aisnps_directory:
    aisnps_directory = data_directory.joinpath("aisnps")
