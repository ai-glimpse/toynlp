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
