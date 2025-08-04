import torch
from datasets import DatasetDict, load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from toynlp.seq2seq.config import DataConfig


def get_dataset(
    dataset_path: str,
    dataset_name: str,
) -> DatasetDict:
    dataset = load_dataset(path=dataset_path, name=dataset_name)
    return dataset  # type: ignore[return-value]


def collate_fn(
    batch: dict[str, list[str]],
    source_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_input = []
    batch_target = []

    for item in batch:
        # TODO: reverse the input text words order(use item["en"][::-1])
        en_tensor = source_tokenizer.encode(item["en"]).ids  # type: ignore[call-arg,index]
        fr_tensor = target_tokenizer.encode(item["fr"]).ids  # type: ignore[call-arg,index]
        batch_input.append(en_tensor)
        batch_target.append(fr_tensor)
    batch_input_tensor = torch.tensor(batch_input, dtype=torch.long)
    batch_target_tensor = torch.tensor(batch_target, dtype=torch.long)
    return batch_input_tensor, batch_target_tensor


def get_split_dataloader(
    dataset: DatasetDict,
    split: str,
    source_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    data_config: DataConfig,
) -> DataLoader:
    dataloader = DataLoader(
        dataset=dataset[split],  # type: ignore[arg-type]
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        shuffle=data_config.shuffle,
        collate_fn=lambda batch: collate_fn(batch, source_tokenizer, target_tokenizer),
        drop_last=True,
    )
    return dataloader


if __name__ == "__main__":
    from toynlp.seq2seq.config import DataConfig, DatasetConfig
    from toynlp.seq2seq.tokenizer import Seq2SeqTokenizer

    data_config = DataConfig()
    dataset_config = DatasetConfig()

    dataset = get_dataset(
        dataset_path=dataset_config.path,
        dataset_name=dataset_config.name,
    )

    source_tokenizer = Seq2SeqTokenizer(lang="en").load()
    target_tokenizer = Seq2SeqTokenizer(lang="fr").load()
    train_dataloader = get_split_dataloader(dataset, "train", source_tokenizer, target_tokenizer, data_config)
    for batch_input, batch_target in train_dataloader:
        print(batch_input.shape, batch_target.shape)
        break
