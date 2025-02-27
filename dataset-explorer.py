import marimo

__generated_with = "0.11.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import plotly.graph_objects as go
    import torch

    from src.lattice.ca import BaseModel
    from src.lattice.ca.rule_224 import Rule_224
    from src.utils import discrete_heatmap
    return BaseModel, Rule_224, discrete_heatmap, go, mo, np, pl, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Defining the data

        **First steps:** define a simple 2-D lattice and train a transfomer architecture on that

        ## Options
        + Langevin dynamics
        + continuous-time Monte Carlo ✅
        """
    )
    return


@app.cell
def _(BaseModel, torch):
    ca = BaseModel(shape=(150, 150), dtype=torch.int8)
    return (ca,)


@app.cell
def _(ca, torch):
    ts = torch.stack([next(ca).lattice for _ in range(250)], axis=0)
    time_series = ts.numpy()
    return time_series, ts


@app.cell
def _(discrete_heatmap, time_series):
    discrete_heatmap(time_series)
    return


if __name__ == "__main__":
    app.run()
