# GPT



## Pre-Training

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
