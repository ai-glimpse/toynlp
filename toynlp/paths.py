import pathlib

_MODEL_PATH = pathlib.Path(__file__).parents[1] / "models"

_MODEL_PATH.mkdir(parents=True, exist_ok=True)


# Word2Vec paths
W2V_TOKENIZER_PATH = _MODEL_PATH / "word2vec" / "tokenizer.json"
CBOW_MODEL_PATH = _MODEL_PATH / "word2vec" / "cbow_model.pt"
SKIP_GRAM_MODEL_PATH = _MODEL_PATH / "word2vec" / "skip_gram_model.pt"


# Seq2Seq paths
SEQ2SEQ_TOKENIZER_EN_PATH = _MODEL_PATH / "seq2seq" / "tokenizer_en.json"
SEQ2SEQ_TOKENIZER_FR_PATH = _MODEL_PATH / "seq2seq" / "tokenizer_fr.json"
SEQ2SEQ_MODEL_PATH = _MODEL_PATH / "seq2seq" / "seq2seq_model.pt"
