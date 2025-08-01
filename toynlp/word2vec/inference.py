import torch
from tokenizers import Tokenizer

from toynlp.device import current_device
from toynlp.word2vec.config import Word2VecPathConfig
from toynlp.word2vec.model import Word2VecModel
from toynlp.word2vec.tokenizer import Word2VecTokenizer


def load_tokenizer_model() -> tuple[Tokenizer, Word2VecModel]:
    path_config = Word2VecPathConfig()
    word2vec_model_path = path_config.model_path
    tokenizer_model_path = path_config.tokenizer_path

    tokenizer = Word2VecTokenizer(tokenizer_model_path).load()
    model = torch.load(str(word2vec_model_path), weights_only=False)
    model.to(current_device)
    model.eval()
    return tokenizer, model


def word_to_vec(word: str) -> torch.Tensor:
    tokenizer, model = load_tokenizer_model()
    token_ids = tokenizer.encode(word).ids
    token_ids_tensor = torch.tensor(token_ids).unsqueeze(0).to(current_device)

    with torch.no_grad():
        vec = model.embedding(token_ids_tensor)
    return vec[0, 0, :]


def vocab_embedding(
    tokenizer: Tokenizer | None = None, model: Word2VecModel | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Returns vocabulary embeddings and corresponding token IDs."""
    if tokenizer is None or model is None:
        tokenizer, model = load_tokenizer_model()

    vocab = tokenizer.get_vocab()
    vocab_ids = list(vocab.values())
    vocab_ids_tensor = torch.tensor(vocab_ids, dtype=torch.long).unsqueeze(0).to(current_device)

    with torch.no_grad():
        embeddings = model.embedding(vocab_ids_tensor).squeeze(0)

    return embeddings, vocab_ids


def find_similar_words(word: str, top_k: int = 5) -> list[str]:
    word_vec = word_to_vec(word)
    return find_similar_words_by_vec(word_vec, top_k)


def calc_vecs_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    return torch.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()


def find_similar_words_by_vec(word_vec: torch.Tensor, top_k: int = 5) -> list[str]:
    tokenizer, model = load_tokenizer_model()
    all_embeddings, vocab_ids = vocab_embedding(tokenizer, model)

    similarities = torch.nn.functional.cosine_similarity(word_vec.unsqueeze(0), all_embeddings, dim=1)
    top_k_indices = torch.topk(similarities, k=top_k).indices

    # Map indices back to actual token IDs and decode them
    similar_token_ids = [vocab_ids[i] for i in top_k_indices]
    similar_words = [tokenizer.decode([token_id]) for token_id in similar_token_ids]
    return similar_words


def evaluate_model_context(text: str) -> None:
    tokenizer, model = load_tokenizer_model()
    token_ids = tokenizer.encode(text).ids
    token_ids_tensor = torch.tensor(token_ids).unsqueeze(0).to(current_device)
    with torch.no_grad():
        print(token_ids_tensor.shape)
        logits = model(token_ids_tensor)
        pred = torch.argmax(logits, dim=1)
        print(tokenizer.decode(pred.tolist()))


def evaludate_embedding() -> None:
    print(f"Embedding for 'machine': {word_to_vec('machine').shape}")

    embeddings, _ = vocab_embedding()
    print("Vocabulary embeddings shape:", embeddings.shape)


def evaluate_similar_words(word: str, top_k: int = 5) -> None:
    similar_words = find_similar_words(word, top_k)
    print(f"[{word}]'s similar words: {similar_words}")


def evaluate_king_queen():
    # king - man + women
    king_vec = word_to_vec("king")
    man_vec = word_to_vec("man")
    woman_vec = word_to_vec("woman")
    res_vec = king_vec - man_vec + woman_vec
    queen_vec = word_to_vec("queen")

    similar_words = find_similar_words_by_vec(res_vec, top_k=10)
    print(f"[king-man+women]'s similar words: {similar_words}")

    print(f"similarity king-queen: {calc_vecs_similarity(king_vec, queen_vec)}")
    print(f"similarity man-woman: {calc_vecs_similarity(man_vec, woman_vec)}")
    print(f"similarity (king-man+women)-queen: {calc_vecs_similarity(res_vec, queen_vec)}")


if __name__ == "__main__":
    # evaluate_model_context("machine learning a method")

    evaludate_embedding()
    evaluate_similar_words("home", top_k=10)
    evaluate_king_queen()
