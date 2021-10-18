import pytest

from pathlib import Path
from ezancestry.commands import predict

tests_directory = Path(__file__).parent
tests_data_directory = tests_directory.joinpath("data")
data_directory = tests_directory.parent.joinpath("data")


# need to test umap, but getting bytecode error
# https://github.com/lmcinnes/umap/issues/473#issuecomment-675340411
@pytest.mark.parametrize("algorithm", [None, "pca", "umap"])
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
        thousand_genomes_directory=None,
        samples_directory=None,
        algorithm=algorithm,
        aisnps_set=None,
    )
    assert (
        results.loc[
            "sample_genome_snps.txt", "predicted_population_superpopulation"
        ]
        == "EUR"
    )
