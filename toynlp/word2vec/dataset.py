import torch
from datasets import DatasetDict, load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from toynlp.word2vec.config import DataConfig


def get_dataset(
    dataset_path: str,
    dataset_name: str,
) -> DatasetDict:
    dataset = load_dataset(path=dataset_path, name=dataset_name)
    return dataset  # type: ignore[return-value]


def collate_cbow_fn(
    batch: dict[str, list[str]],
    tokenizer: Tokenizer,
    data_config: DataConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    cbow_n_words = data_config.cbow_n_words
    batch_input = []
    batch_target = []

    for line in batch:
        token_ids = tokenizer.encode(line["text"]).ids  # type: ignore[call-arg,index]
        if len(token_ids) < cbow_n_words * 2 + 1:
            continue

        for i in range(cbow_n_words, len(token_ids) - cbow_n_words):
            context = token_ids[i - cbow_n_words : i] + token_ids[i + 1 : i + cbow_n_words + 1]
            target = token_ids[i]
            batch_input.append(context)
            batch_target.append(target)
    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_target = torch.tensor(batch_target, dtype=torch.long)
    return batch_input, batch_target


def get_split_dataloader(
    dataset: DatasetDict,
    split: str,
    tokenizer: Tokenizer,
    data_config: DataConfig,
) -> DataLoader:
    dataloader = DataLoader(
        dataset=dataset[split],  # type: ignore[arg-type]
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        shuffle=data_config.shuffle,
        collate_fn=lambda batch: collate_cbow_fn(batch, tokenizer, data_config),
        drop_last=True,
    )
    return dataloader


if __name__ == "__main__":
    from toynlp.word2vec.config import DataConfig, DatasetConfig, Word2VecPathConfig
    from toynlp.word2vec.tokenizer import Word2VecTokenizer

    data_config = DataConfig()
    dataset_config = DatasetConfig()
    path_config = Word2VecPathConfig()

    dataset = get_dataset(
        dataset_path=dataset_config.path,
        dataset_name=dataset_config.name,
    )

    tokenizer = Word2VecTokenizer(model_path=path_config.tokenizer_path).load()
    train_dataloader = get_split_dataloader(dataset, "train", tokenizer, data_config)
    for batch_input, batch_target in train_dataloader:
        print(batch_input.shape, batch_target.shape)
        break
