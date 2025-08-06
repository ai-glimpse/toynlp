# Sequence-to-Sequence (Seq2Seq) Model Tasks
seq2seq-tokenize:
    uv run python toynlp/seq2seq/tokenizer.py

seq2seq-train:
    uv run python toynlp/seq2seq/train.py \
        --config configs/seq2seq/default.yml

seq2seq-infer:
    uv run python toynlp/seq2seq/inference.py

seq2seq-eval:
    uv run python toynlp/seq2seq/evaluation.py

seq2seq-train-eval: seq2seq-train seq2seq-eval
