from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader


def get_dataset(
    dataset_path: str = "Salesforce/wikitext",
    dataset_name: str = "wikitext-2-raw-v1",
    ) -> DatasetDict:
    dataset = load_dataset(path=dataset_path, name=dataset_name)
    return dataset  # type: ignore[return-value]


def get_split_dataloader(
    dataset: DatasetDict,
    split: str,
    batch_size: int = 32,
) -> DataLoader:
    dataloader = DataLoader(
        dataset=dataset[split],  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=True,
        )
    return dataloader


if __name__ == "__main__":
    dataset = get_dataset()
    train_dataloader = get_split_dataloader(dataset, "train")
    for batch in train_dataloader:
        print(batch)
        break
