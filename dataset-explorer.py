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

    from src.lattice.ca import Rule224
    from src.utils import discrete_heatmap
    return Rule224, discrete_heatmap, go, mo, np, pl, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Defining the data

        **First steps:** define a simple 2-D lattice and train a transfomer architecture   
        **Potential route:** study on a 3-D lattice

        ## exploration strategy

        + [x] Cellular Automata
        + [ ] continuous Cellular Automata
        + [ ] Lenia *(discretized biology-mimicking CA)*
        + [ ] continuous-time Monte Carlo ✅
        + [ ] Langevin dynamics
        """
    )
    return


@app.cell
def _(Rule224, discrete_heatmap, torch):
    ca = Rule224(shape=(150, 150), dtype=torch.int8)
    time_series = ca.time_steps(128)
    discrete_heatmap(time_series)
    return ca, time_series


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
