from toynlp.nnlm.model import NNLM
import torch


def test_nnlm_model_architecture():
    n = 6
    vocab_size = 17964
    m = 100
    h = 60
    model = NNLM(seq_len=n, vocab_size=vocab_size, embedding_dim=m, hidden_dim=h)
    # |V |(1 + nm + h) + h(1 + (n − 1)m)
    assert (
        sum(p.numel() for p in model.parameters())
        == vocab_size * (1 + n * m + h) + h * (1 + (n - 1) * m)
    )
    assert model(torch.randint(0, vocab_size, (2, 5))).shape == torch.Size([2, vocab_size])
