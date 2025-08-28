from datasets import DatasetDict, load_dataset



def get_dataset(
    dataset_path: str,
    dataset_name: str | None,
) -> DatasetDict:
    dataset = load_dataset(path=dataset_path, name=dataset_name)
    return dataset  # type: ignore[return-value]


def dataset_transform(dataset: DatasetDict) -> DatasetDict:
    # Apply any necessary transformations to the dataset here
    return dataset



if __name__ == "__main__":
    pass
