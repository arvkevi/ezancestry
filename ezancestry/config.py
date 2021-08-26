from pathlib import Path

data_directory = Path(__file__).parents[1].joinpath("data")
models_directory = data_directory.joinpath("models")
aisnps_directory = data_directory.joinpath("aisnps")
samples_directory = data_directory.joinpath("samples")



config = ConfigParser()
config.read("conf.ini")
census_api_key = config["data.downloader"]["census_api_key"]