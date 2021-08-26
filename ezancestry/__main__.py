import fire

from ezancestry.aisnps import generate_aisnps

# TODO User inputs are:
# 23andMe (or other provider) file
# Directory of 23andMe files
# Pandas DataFrame or some other object?


def main():
    fire.Fire(
        {
            "generate-aisnps": generate_aisnps,
            "train": train_model,
            "predict": predict_ancestry,
            "plot": plot,
        }
    )


if __name__ == "__main__":
    main()
