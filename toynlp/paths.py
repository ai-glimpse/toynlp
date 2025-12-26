import pathlib

_MODEL_PATH = pathlib.Path(__file__).parents[1] / "models"
# _MODEL_PATH = pathlib.Path("/data/models")
_MODEL_PATH.mkdir(parents=True, exist_ok=True)


# Word2Vec paths
W2V_TOKENIZER_PATH = _MODEL_PATH / "word2vec" / "tokenizer.json"
CBOW_MODEL_PATH = _MODEL_PATH / "word2vec" / "cbow_model.pt"
SKIP_GRAM_MODEL_PATH = _MODEL_PATH / "word2vec" / "skip_gram_model.pt"


# Seq2Seq paths
SEQ2SEQ_TOKENIZER_PATH_MAP = {
    "en": _MODEL_PATH / "seq2seq" / "tokenizer_en.json",
    "de": _MODEL_PATH / "seq2seq" / "tokenizer_de.json",
}
SEQ2SEQ_MODEL_PATH = _MODEL_PATH / "seq2seq" / "seq2seq_model.pt"

# Attention paths
ATTENTION_TOKENIZER_PATH_MAP = {
    "en": _MODEL_PATH / "attention" / "tokenizer_en.json",
    "de": _MODEL_PATH / "attention" / "tokenizer_de.json",
}
ATTENTION_MODEL_PATH = _MODEL_PATH / "attention" / "attention_model.pt"


# Fasttext paths
FASTTEXT_TOKENIZER_PATH = _MODEL_PATH / "fasttext" / "tokenizer.json"
FASTTEXT_MODEL_PATH = _MODEL_PATH / "fasttext" / "fasttext_model.pt"

# Transformer paths
TRANSFORMER_TOKENIZER_PATH = _MODEL_PATH / "transformer" / "tokenizer.json"
TRANSFORMER_MODEL_PATH = _MODEL_PATH / "transformer" / "transformer_model.pt"

# GPT paths
GPT_TOKENIZER_PATH = _MODEL_PATH / "gpt" / "tokenizer.json"
GPT_MODEL_PATH = _MODEL_PATH / "gpt" / "gpt_model.pt"
GPT_SFT_MODEL_PATH = _MODEL_PATH / "gpt" / "gpt_sft_model.pt"
GPT_ARC_MODEL_PATH = _MODEL_PATH / "gpt" / "gpt_arc_model.pt"
GPT_ARC_RL_MODEL_PATH = _MODEL_PATH / "gpt" / "gpt_arc_rl_model.pt"
SST2GPT_MODEL_PATH = _MODEL_PATH / "gpt" / "sst2_gpt_model.pt"


# Bert paths
BERT_TOKENIZER_PATH = _MODEL_PATH / "bert" / "tokenizer.json"
BERT_MODEL_PATH = _MODEL_PATH / "bert" / "bert_model.pt"
SST2BERT_MODEL_PATH = _MODEL_PATH / "bert" / "sst2_bert_model.pt"
