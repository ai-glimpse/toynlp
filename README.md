

[![Python](https://img.shields.io/pypi/pyversions/toynlp.svg?color=%2334D058)](https://pypi.org/project/toynlp/)
[![PyPI](https://img.shields.io/pypi/v/toynlp?color=%2334D058&label=pypi%20package)](https://pypi.org/project/toynlp/)
[![PyPI Downloads](https://static.pepy.tech/badge/toynlp)](https://pepy.tech/projects/toynlp)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

[![Build Docs](https://github.com/ai-glimpse/toynlp/actions/workflows/build_docs.yaml/badge.svg)](https://github.com/ai-glimpse/toynlp/actions/workflows/build_docs.yaml)
[![Test](https://github.com/ai-glimpse/toynlp/actions/workflows/test.yaml/badge.svg)](https://github.com/ai-glimpse/toynlp/actions/workflows/test.yaml)
[![Codecov](https://codecov.io/gh/ai-glimpse/toynlp/branch/master/graph/badge.svg)](https://codecov.io/gh/ai-glimpse/toynlp)
[![GitHub License](https://img.shields.io/github/license/ai-glimpse/toynlp)](https://github.com/ai-glimpse/toynlp/blob/master/LICENSE)


# ToyNLP

Implementing classic NLP models from scratch with clean code and easy-to-understand architecture.

> This library is for educational purposes only. It is not optimized for production use.
> And it may contain bugs CURRENTLY, so feel free to contribute and report issues.
>
> Until now, we only do simple tests, which is not enough. But we will do much more rigorous testing in the FUTURE.
> And we will add more docs for you can RUN it easily, add more playgrounds for you to experiment with the models and look inside the model implementations.


## Models

10 important NLP models range from 2003 to 2020:

- [x] NNLM(2003)
- [x] Word2Vec(2013)
- [x] Seq2Seq(2014)
- [x] Attention(2015)
- [x] fastText(2016)
- [x] Transformer(2017)
- [ ] BERT(2018)
- [ ] GPT(2018)
- [ ] XLNet(2019)
- [ ] T5(2020)



## FAQ

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
