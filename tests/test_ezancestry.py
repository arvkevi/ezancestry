import pytest

from pathlib import Path
from ezancestry.commands import predict

tests_directory = Path(__file__).parent
tests_data_directory = tests_directory.joinpath("data")
data_directory = tests_directory.parent.joinpath("data")

@pytest.mark.parametrize("algorithm", [None, "pca"])
def test_predict_from_file(algorithm):
    """Test the predict function from file."""
    sample_filename = tests_data_directory.joinpath("sample_genome_snps.txt")
    results = predict(
        input_data=sample_filename,
        output_directory=None,
        write_predictions=False,
        models_directory=data_directory.joinpath("models"),
        aisnps_directory=None,
        n_components=None,
        k=None,
        algorithm=algorithm,
        aisnps_set=None,
    )
    assert (
        results.loc[
            "sample_genome_snps.txt", "predicted_population_superpopulation"
        ]
        == "EUR"
    )
