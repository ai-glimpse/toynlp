# Dev
test:
    uv run pytest --doctest-modules -v --cov=toynlp --cov-fail-under 0 --cov-report=term --cov-report=xml --cov-report=html toynlp tests


# FastText model
fasttext-tokenize:
    uv run python toynlp/fasttext/tokenizer.py

fasttext-train *args:
    uv run python toynlp/fasttext/train.py {{args}}

fasttext-eval:
    uv run python toynlp/fasttext/evaluation.py

# Attention-based Seq2Seq Model Tasks
attention-train *args:
    uv run python toynlp/attention/train.py {{args}}

attention-infer:
    uv run python toynlp/attention/inference.py

attention-eval:
    uv run python toynlp/attention/evaluation.py

attention-train-eval: attention-train attention-eval


# Sequence-to-Sequence (Seq2Seq) Model Tasks
seq2seq-tokenize:
    uv run python toynlp/seq2seq/tokenizer.py

seq2seq-train *args:
    uv run python toynlp/seq2seq/train.py {{args}}

seq2seq-infer:
    uv run python toynlp/seq2seq/inference.py

seq2seq-eval:
    uv run python toynlp/seq2seq/evaluation.py

seq2seq-train-eval: seq2seq-train seq2seq-eval


# Transformer Model Tasks
transformer-tokenize:
    uv run python toynlp/transformer/tokenizer.py

transformer-train *args:
    uv run python toynlp/transformer/train.py {{args}}

transformer-infer:
    uv run python toynlp/transformer/inference.py

transformer-eval:
    uv run python toynlp/transformer/evaluation.py

transformer-train-eval: transformer-train transformer-eval


# Bert Model Tasks
bert-tokenize:
    uv run python toynlp/bert/tokenizer.py

bert-train *args:
    uv run python toynlp/bert/train.py {{args}}

bert-infer:
    uv run python toynlp/bert/inference.py

bert-eval:
    uv run python toynlp/bert/evaluation.py

bert-train-eval: bert-train bert-eval


# GPT Model Tasks
gpt-tokenize:
    uv run python toynlp/gpt/tokenizer.py

gpt-train *args:
    uv run python toynlp/gpt/train.py {{args}}
