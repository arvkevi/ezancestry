import fire

from ezancestry.aisnps import generate_aisnps


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
