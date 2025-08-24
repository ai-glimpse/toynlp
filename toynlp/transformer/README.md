# Transformer

A PyTorch implementation of Transformer model based on "Attention Is All You Need" paper.

## Result

We got test bleu score of 30.07 at [bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k).


## The mistakes that I made

### MHA: split multi-head in a wrong way

The wrong impl:

```python
q = q.view(batch_size, self.config.head_num, seq_length, q_k_head_dim)
k = k.view(batch_size, self.config.head_num, seq_length, q_k_head_dim)
v = v.view(batch_size, self.config.head_num, seq_length, v_head_dim)
```

The right impl:

```python
q = q.view(batch_size, q.size(1), self.config.head_num, q_k_head_dim).transpose(1, 2)
k = k.view(batch_size, k.size(1), self.config.head_num, q_k_head_dim).transpose(1, 2)
v = v.view(batch_size, v.size(1), self.config.head_num, v_head_dim).transpose(1, 2)
```

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [hyunwoongko/transformer](https://github.com/hyunwoongko/transformer)
