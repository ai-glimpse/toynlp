# GPT



## Pre-Training


### The results

The differences with the original GPT model:

| Aspect | Original GPT | This Implementation |
|:--------:|:---------------:|:-------------------:|
| Training Epochs | 100 | 27 |

Performance comparison:

| Metric | Original GPT | This Implementation |
|:--------:|:---------------:|:-------------------:|
| Perplexity| 18.4 | 25.27|
| SST2 Accuracy | 91.3% | **91.7%** |


### The dataset is around 800M words(1B tokens)

We use BooksCorpus dataset for pre-training.

The BERT paper claims that BooksCorpus has 800M words:
> For the pre-training corpus we use the BooksCorpus (800M words) (Zhu et al.,2015) and ...

The GPT paper claims that BooksCorpus has around 1B tokens:
> Unsupervised pre-training We use the BooksCorpus dataset [71] for training the language model.
> It contains over 7,000 unique unpublished books ... An alternative dataset, the 1B Word Benchmark ... is approximately the same size.

And when we train our GPT model on BooksCorpus, the batch size is 24, sequence length is 512, and one epoch contains 116599 steps, so the total token count is:
```
24(batch size) * 512(sequence length) * 116599(steps per epoch) = 143265792 tokens = 1.43B tokens
```
The extra tokens may come from different tokenization processes.


### The model architecture should be correct

The model has 116.5M(116534784) parameters, Which is EXACTLY the same as huggingface transformers' gpt model(`AutoModelForCausalLM.from_pretrained("openai-community/openai-gpt")`) parameter count.


### The training process should be correct

We can overfit a small dataset

```bash
====================================================================================================
[TRAIN] Sample Predictions:
Input tokens: <bos> | # | # | 1 | God | – | Poems | on | God | ,
Target tokens: # | # | 1 | God | – | Poems | on | God | , | Creator
Predicted tokens: # | # | 1 | God | – | Poems | on | God | , | Creator
====================================================================================================
Epoch 758/1000 - Train Loss: 0.0003, Train Perplexity: 1.0003, LR: 0.000100,
====================================================================================================
```

## Supervised Fine-Tuning(LoRA)

### The results

> We can get better results sometimes...
ARC-Challenge: 24.91% (292/1172)
ARC-Easy: 25.59% (608/2376)
Overall: 25.37% (900/3548)



## The mistakes that I made


### Padding mask broadcasted on the wrong axis

**Symptom.** Long runs would suddenly see the loss jump to `nan` right after a batch containing padding-only suffixes. Inspecting the attention logits right before `softmax` showed entire rows equal to `-inf`.

```python
import torch
logits = torch.tensor([
	[0.31, 0.12, -0.44, -1.18],
	[0.48, -0.27, -0.73, -0.95],
	[-inf, -inf, -inf, -inf],
	[-inf, -inf, -inf, -inf],
])
```

**How the bug happened.** I built the padding mask as

```python
pad_mask_bad = (input_ids != padding_idx).unsqueeze(1).unsqueeze(3)
```

which yields shape `(batch, 1, seq_len, 1)`. For a toy batch with padding at the end:

```python
ids = torch.tensor([[5, 7, 0, 0]])
pad_mask_bad[0, 0] == tensor([
                            [ True],
                            [ True],
                            [False],
                            [False]])
```

During attention this mask must broadcast to `(batch, heads, seq_q, seq_k)`. The third query row owns only a single `False`, so broadcasting replicates it across every key position:

```python
row_2_after_broadcast = [False, False, False, False]
```

After applying the causal mask everything in that row stays `False`, so the logits become `-inf`, and `softmax` turns the row into `nan`. The failure only appeared when an entire suffix was padding, which explains why it slipped through basic smoke tests.

**Step-by-step view.**

1. Build the naïve mask: `pad_mask_bad.shape == (1, 1, 4, 1)`.
2. Combine with the lower-triangular causal mask:
```python
[[ True, False, False, False],
    [ True,  True, False, False],
    [False, False, False, False],
    [False, False, False, False]]
```
3. Apply to logits → rows 2 and 3 contain only `-inf` → `nan` attention weights.

**The fix.** Keep the key axis explicit:

```python
pad_mask_good = (input_ids != padding_idx).unsqueeze(1).unsqueeze(2)
```

Now the mask starts at `(batch, 1, 1, seq_len)` and broadcasting preserves the column-wise padding information:

```python
[[ True, False, False, False],
 [ True,  True, False, False],
 [ True,  True, False, False],
 [ True,  True, False, False]]
```

Rows 2 and 3 still attend to the earlier valid tokens, so the logits stay finite and the model trains normally.

**Lessons learned.** Masks are just tensors, so broadcast semantics matter. Printing the exact shapes before and after each operation (or writing a quick unit test) is a cheap way to catch mistakes that otherwise only show up hours into training.


### We don't add a special token for end of sentence

This makes the supervised fine-tuning task harder, because the model has to predict the end of sentence by itself.

For continue the sft, we choose to use `___` as the end of sentence token temporarily.
