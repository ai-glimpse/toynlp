import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Transformer Model Architecture""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Encoder""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Decoder""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Positional Encoding


    ### Why

    - Why use positional encoding?
    >Since our model contains no recurrence and no convolution, in order for the model to **make use of the
    order of the sequence**, we must **inject** some information about the **relative or absolute position** of the
    tokens in the sequence.

    - Why use sinusoid functions?
    >Wechose this function because we **hypothesized** it would allow the model to easily learn to attend by
    relative positions, **since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$**.

    ### What
    $$
    PE_{(\text{pos}, 2i)} = sin\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)
    $$

    $$
    PE_{(\text{pos}, 2i+1)} = cos\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)
    $$

    The inputs:

    - $\text{pos}$: the position in the sequence, up to `max_length`
    - $i$: the dimension, up to `d_model`
    - $d_{\text{model}}$: the model dimension(the dimension of the embedding)

    The output

    - $PE$: the positional encoding, which is a matrix of shape `(max_length, d_model)`


    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    ### Details

    > The wavelengths form a geometric progression from $2\pi$ to $10000 · 2\pi$.

    According to the [Sinusoid form](https://en.wikipedia.org/wiki/Sine_wave), we have:

    $$
    y(t) = Asin(\omega t + \phi) = Asin(2\pi ft + \phi)
    $$

    We can simplify it into this format in transformer's symbols:

    $$
    y(pos, 2i) = sin(2\pi f_{2i} \cdot pos) = sin(\frac{pos}{10000^{2i/d_{\text{model}}}})
    $$

    where the $f_{2i}$ is the frequency in dimension $2i$. We can also derive that the format of $f_{2i}$:

    $$
    f_{2i} = \frac{1}{2\pi  \cdot 10000^{2i/d_{\text{model}}}}
    $$

    $$
    f_i = \frac{1}{2\pi \cdot 10000^{i/d_{\text{model}}}}
    $$

    Note the transformation of wavelength $\lambda$ and frequency $f$:

    $$
    f = \frac{1}{\lambda}
    $$


    It means that for dimension $i$, the wavelength is:

    $$
    \lambda_i = \frac{1}{f_i} = 10000^{i/d_{\text{model}}}  \cdot 2\pi
    $$

    When $i$ range from 0 to $d_{model}-1$, the $10000^{i/d_{\text{model}}}$ range from $1$ to $10000$.
    So the wavelength $\lambda_i$ range from $2\pi$ to $10000 \cdot 2\pi$.

    """
    )
    return


@app.cell
def _():
    d_model = 512
    # Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens
    max_length = 25000
    return d_model, max_length


@app.cell
def _(d_model, max_length, torch):
    embedding = torch.zeros(max_length, d_model)
    pos = torch.arange(max_length).unsqueeze(1)
    i = torch.arange(d_model).unsqueeze(0)

    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
    #   # shape: (max_length, d_model)
    angle_rads = pos * angle_rates

    pe = torch.zeros_like(angle_rads)
    pe[:, 0::2] = torch.sin(angle_rads[:, 0::2])  # apply sin to even indices in the array; 2i
    pe[:, 1::2] = torch.cos(angle_rads[:, 1::2])  # apply cos to odd
    return angle_rads, pe


@app.cell
def _(d_model, pe, plt):
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(pe, cmap="plasma")
    plt.xlabel("Dimention")
    plt.xlim((0, d_model))
    plt.ylabel("Position")
    plt.colorbar()
    plt.title("Positional Encoding")
    plt.show()
    return


@app.cell
def _(angle_rads):
    angle_rads.min()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
