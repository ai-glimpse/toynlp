# Dev
test:
    uv run pytest --doctest-modules -v --cov=toynlp --cov-fail-under 0 --cov-report=term --cov-report=xml --cov-report=html toynlp tests


# Attention-based Seq2Seq Model Tasks
attention-train:
    uv run python toynlp/attention/train.py

attention-infer:
    uv run python toynlp/attention/inference.py

attention-eval:
    uv run python toynlp/attention/evaluation.py

attention-train-eval: attention-train attention-eval


# Sequence-to-Sequence (Seq2Seq) Model Tasks
seq2seq-tokenize:
    uv run python toynlp/seq2seq/tokenizer.py \
        --config configs/seq2seq/default.yml

seq2seq-train:
    uv run python toynlp/seq2seq/train.py \
        --config configs/seq2seq/default.yml

seq2seq-infer:
    uv run python toynlp/seq2seq/inference.py

seq2seq-eval:
    uv run python toynlp/seq2seq/evaluation.py

seq2seq-train-eval: seq2seq-train seq2seq-eval
