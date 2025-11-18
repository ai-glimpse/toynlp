

[![Python](https://img.shields.io/pypi/pyversions/toynlp.svg?color=%2334D058)](https://pypi.org/project/toynlp/)
[![PyPI](https://img.shields.io/pypi/v/toynlp?color=%2334D058&label=pypi%20package)](https://pypi.org/project/toynlp/)
[![PyPI Downloads](https://static.pepy.tech/badge/toynlp)](https://pepy.tech/projects/toynlp)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

[![Build Docs](https://github.com/ai-glimpse/toynlp/actions/workflows/build_docs.yaml/badge.svg)](https://github.com/ai-glimpse/toynlp/actions/workflows/build_docs.yaml)
[![Test](https://github.com/ai-glimpse/toynlp/actions/workflows/test.yaml/badge.svg)](https://github.com/ai-glimpse/toynlp/actions/workflows/test.yaml)
[![Codecov](https://codecov.io/gh/ai-glimpse/toynlp/branch/master/graph/badge.svg)](https://codecov.io/gh/ai-glimpse/toynlp)
<!-- [![GitHub License](https://img.shields.io/github/license/ai-glimpse/toynlp)](https://github.com/ai-glimpse/toynlp/blob/master/LICENSE) -->


# ToyNLP

Implementing classic NLP models from scratch with clean code and easy-to-understand architecture.

> This library is for educational purposes only. It is not optimized for production use.
> And it may contain bugs CURRENTLY, so feel free to contribute and report issues.
>
> Until now, we only do simple tests, which is not enough. But we will do much more rigorous testing in the FUTURE.
> And we will add more docs for you can RUN it easily, add more playgrounds for you to experiment with the models and look inside the model implementations.


## Models

8 important NLP models range from 2003 to 2020:

| Model & Paper | Code | Doc(EN) | Blog(ZH) |
|:--------------|:----:|:--------:|:--------:|
| NNLM(2003)<br>[![JMLR](https://img.shields.io/badge/JMLR-Volume%203-blue.svg)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)  | ✅ | ![Coming soon](https://img.shields.io/badge/Doc-Coming%20soon-lightgrey.svg) | ![Coming soon](https://img.shields.io/badge/Blog-Coming%20soon-lightgrey.svg) |
| Word2Vec(2013)<br>[![arXiv](https://img.shields.io/badge/arXiv-1301.3781-b31b1b.svg)](https://arxiv.org/abs/1301.3781) | ✅ | ![Coming soon](https://img.shields.io/badge/Doc-Coming%20soon-lightgrey.svg) | ![Coming soon](https://img.shields.io/badge/Blog-Coming%20soon-lightgrey.svg) |
| Seq2Seq(2014)<br>[![arXiv](https://img.shields.io/badge/arXiv-1409.3215-b31b1b.svg)](https://arxiv.org/abs/1409.3215) | ✅ | ![Coming soon](https://img.shields.io/badge/Doc-Coming%20soon-lightgrey.svg) | ![Coming soon](https://img.shields.io/badge/Blog-Coming%20soon-lightgrey.svg) |
| Attention(2014)<br>[![arXiv](https://img.shields.io/badge/arXiv-1409.0473-b31b1b.svg)](https://arxiv.org/abs/1409.0473) | ✅ | ![Coming soon](https://img.shields.io/badge/Doc-Coming%20soon-lightgrey.svg) | ![Coming soon](https://img.shields.io/badge/Blog-Coming%20soon-lightgrey.svg) |
| fastText(2016)<br>[![arXiv](https://img.shields.io/badge/arXiv-1607.04606-b31b1b.svg)](https://arxiv.org/abs/1607.04606) | ✅ | ![Coming soon](https://img.shields.io/badge/Doc-Coming%20soon-lightgrey.svg) | ![Coming soon](https://img.shields.io/badge/Blog-Coming%20soon-lightgrey.svg) |
| Transformer(2017)<br>[![arXiv](https://img.shields.io/badge/arXiv-1706.03762-b31b1b.svg)](https://arxiv.org/abs/1706.03762) | ✅ | ![Coming soon](https://img.shields.io/badge/Doc-Coming%20soon-lightgrey.svg) | ![Coming soon](https://img.shields.io/badge/Blog-Coming%20soon-lightgrey.svg) |
| GPT(2018)<br>[![OpenAI](https://img.shields.io/badge/OpenAI-Paper%20-00A67E.svg)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | ✅ | [![Doc](https://img.shields.io/badge/Doc-GPT-green.svg)](./toynlp/gpt/README.md) | ![Coming soon](https://img.shields.io/badge/Blog-Coming%20soon-lightgrey.svg) |
| BERT(2018)<br>[![arXiv](https://img.shields.io/badge/arXiv-1810.04805-b31b1b.svg)](https://arxiv.org/abs/1810.04805) | ✅ | [![Doc](https://img.shields.io/badge/Doc-BERT-green.svg)](./toynlp/bert/README.md) | [![Blog](https://img.shields.io/badge/Blog-BERT-orange.svg)](https://datahonor.com/blog/2025/11/02/bert/) |



## FAQ

### I see there are 10 models before, why only 8 now?

Yes, I removed XLNet and T5. The main reason is that I don't have so many GPUs to train these models. Without training them by myself, I can't guarantee the correctness of the implementations.
So I just keep the models that I can train and verify myself.


### I find there are DIFFERENCES between the implementations in toynlp and original papers.

Yes, there are some differences in the implementations. The goal of toynlp is to provide a simple and educational implementation of these models, which may not include all the optimizations and features in original papers.

The reason is that I want to focus on the core ideas and concepts behind each model, rather than getting bogged down in implementation details. Especially when the original papers may introduce complexities that are not essential for understanding the main contributions of the work.

But, I do need to add docs for each model to clarify these differences and provide guidance on how to use the implementations effectively. I'll do this later. Let's first make it work and then make it better.


### Where is GPT-2 and other LLMs?

Well, it's in [toyllm](https://github.com/ai-glimpse/toyllm)!
I separated the models into two libraries, `toynlp` for traditional "small" NLP models and `toyllm` for LLMs, which are typically larger and more complex.

### Like the "toy" style, anything else?

Glad you asked! The "toy" style is all about simplicity and educational value.
We have another two toys besides toynlp and toyllm: [toyml](https://github.com/ai-glimpse/toyml) for traditional machine learning models; [toyrl](https://github.com/ai-glimpse/toyrl) for deep reinforcement learning models.
