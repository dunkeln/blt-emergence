

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    from einops import rearrange
    from src.lattice import Rule224
    from src.utils import heatmap
    import plotly.graph_objects as go
    from src.model import RoPE, EncoderBlock, PatchingTransformer
    return PatchingTransformer, Rule224, heatmap, mo, torch


@app.cell
def _(mo):
    mo.md(r"""## Ideas from references""")
    return


@app.cell
def _(Rule224, torch):
    ca = Rule224(shape=(50, 50), dtype=torch.int8)
    time_series = ca.time_steps(100)
    return (time_series,)


@app.cell
def _(heatmap, time_series):
    heatmap(time_series)
    return


@app.cell
def _(mo):
    mo.md(r"""### Scheme: Patching from Autoregressive Model""")
    return


@app.cell
def _(PatchingTransformer, time_series):
    x = time_series.squeeze(1)

    model = PatchingTransformer()
    logits = model(x)
    return logits, model


@app.cell
def _(logits, torch):
    probs = torch.softmax(logits, dim=-1)
    probs
    return


@app.cell
def _(logits, model):
    model.compute_entropy(logits).size()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
