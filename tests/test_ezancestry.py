from pathlib import Path
from ezancestry.commands import predict

tests_directory = Path(__file__).parent
data_directory = tests_directory.joinpath("data")


def test_predict_from_file():
    """Test the predict function from file."""
    sample_filename = data_directory.joinpath("sample_genome_snps.txt")
    results = predict(
        input_data=sample_filename,
        output_directory=None,
        write_predictions=False,
        models_directory=data_directory,
        aisnps_directory=None,
        n_components=None,
        k=None,
        thousand_genomes_directory=None,
        samples_directory=None,
        algorithm=None,
        aisnps_set=None,
    )
    assert (
        results.loc[
            "sample_genome_snps.txt", "predicted_population_superpopulation"
        ]
        == "EUR"
    )
