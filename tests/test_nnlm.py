import torch

from toynlp.nnlm.config import NNLMConfig
from toynlp.nnlm.model import NNLM


def test_nnlm_model_architecture() -> None:
    n = 6
    vocab_size = 17964
    m = 100
    h = 60
    config = NNLMConfig(
        context_size=n,
        vocab_size=vocab_size,
        embedding_dim=m,
        hidden_dim=h,
        with_dropout=False,
        with_direct_connection=True,
    )
    model = NNLM(config)
    # |V |(1 + nm + h) + h(1 + (n − 1)m)
    assert sum(p.numel() for p in model.parameters()) == vocab_size * (1 + n * m + h) + h * (1 + (n - 1) * m)
    assert model(torch.randint(0, vocab_size, (2, 5))).shape == torch.Size(
        [2, vocab_size],
    )
