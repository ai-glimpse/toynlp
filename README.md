<center>

[![Python](https://img.shields.io/pypi/pyversions/toynlp.svg?color=%2334D058)](https://pypi.org/project/toynlp/)
[![PyPI](https://img.shields.io/pypi/v/toynlp?color=%2334D058&label=pypi%20package)](https://pypi.org/project/toynlp/)
[![PyPI Downloads](https://static.pepy.tech/badge/toynlp)](https://pepy.tech/projects/toynlp)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

[![Build Docs](https://github.com/ai-glimpse/toynlp/actions/workflows/build_docs.yaml/badge.svg)](https://github.com/ai-glimpse/toynlp/actions/workflows/build_docs.yaml)
[![Test](https://github.com/ai-glimpse/toynlp/actions/workflows/test.yaml/badge.svg)](https://github.com/ai-glimpse/toynlp/actions/workflows/test.yaml)
[![Codecov](https://codecov.io/gh/ai-glimpse/toynlp/branch/master/graph/badge.svg)](https://codecov.io/gh/ai-glimpse/toynlp)
[![GitHub License](https://img.shields.io/github/license/ai-glimpse/toynlp)](https://github.com/ai-glimpse/toynlp/blob/master/LICENSE)

</center>

# ToyNLP

NLP models with clean implementation.


## Models

10 important NLP models range from 2003 to 2020:

- [x] NNLM(2003)
- [x] Word2Vec(2013)
- [ ] Seq2Seq(2014)
- [ ] Attention(2015)
- [ ] fastText(2016)
- [ ] Transformer(2017)
- [ ] BERT(2018)
- [ ] GPT(2018)
- [ ] XLNet(2019)
- [ ] T5(2020)

You may wonder where is GPT-2? Well, it's in [toyllm](https://github.com/ai-glimpse/toyllm)!
I separated the models into two libraries, `toynlp` for traditional "small" NLP models and `toyllm` for LLMs, which are typically larger and more complex.
