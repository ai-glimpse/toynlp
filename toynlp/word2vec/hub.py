from pathlib import Path

import torch
from huggingface_hub import HfApi
from tokenizers import Tokenizer

from toynlp.word2vec.config import Word2VecPathConfig
from toynlp.word2vec.model import CbowModel
from toynlp.word2vec.tokenizer import Word2VecTokenizer


def load_tokenizer_model() -> tuple[Tokenizer, CbowModel]:
    path_config = Word2VecPathConfig()
    word2vec_model_path = path_config.model_path
    tokenizer_model_path = path_config.tokenizer_path

    tokenizer = Word2VecTokenizer(tokenizer_model_path).load()
    model = torch.load(str(word2vec_model_path), weights_only=False)
    return tokenizer, model


def push_model_to_hub(model: CbowModel, repo_id: str) -> None:
    model.save_pretrained(repo_id)
    """Push the model to the Hugging Face Hub."""
    model.push_to_hub(repo_id)
    print(f"Model pushed to Hugging Face Hub at {repo_id}")


def push_tokenizer_to_hub(path: Path, repo_id: str) -> None:
    """Push the tokenizer to the Hugging Face Hub."""
    api = HfApi()
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path.name,
        repo_id=repo_id,
    )


if __name__ == "__main__":
    # Load the trained model
    tokenizer, model = load_tokenizer_model()
    # Push the model to the Hugging Face Hub

    repo_id = "AI-Glimpse/word2vec-cbow-wiki-103"
    # push_model_to_hub(model, repo_id)
    push_tokenizer_to_hub(Word2VecPathConfig().tokenizer_path, repo_id)
