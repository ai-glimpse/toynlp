# Word2Vec

We implemented Word2Vec models using PyTorch, inspired by the original Word2Vec [paper](https://arxiv.org/pdf/1301.3781).
The models are trained on Wikipedia data in 103 languages, and we provide both CBOW (Continuous Bag of Words) and Skip-Gram models.


## Models

There are two Word2Vec models and one tokenizer available:

- **CBOW**: Continuous Bag of Words model, trained on Wikipedia 103 languages.
- **Skip-Gram**: Skip-Gram model, trained on Wikipedia 103 languages.
- **Tokenizer**: Tokenizer for CBOW & Skip-Gram, trained on Wikipedia 103 languages.

You can find all the models in the [Hugging Face Hub: AI-Glimpse/word2vec](https://huggingface.co/AI-Glimpse/word2vec).


## References
- https://github.com/OlgaChernytska/word2vec-pytorch
- https://github.com/graykode/nlp-tutorial/tree/master/1-2.Word2Vec
