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

    from src.lattice import Rule224, StochasticLenia, MCSTLattice
    from src.utils.plot import heatmap
    return (
        MCSTLattice,
        Rule224,
        StochasticLenia,
        go,
        heatmap,
        mo,
        np,
        pl,
        torch,
    )


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
        + [x] continuous-time Monte Carlo ✅
        + [ ] Langevin dynamics
        """
    )
    return


@app.cell
def _(Rule224, heatmap, torch):
    ca = Rule224(shape=(40, 40), dtype=torch.int8)
    time_series = ca.time_steps(128).squeeze(dim=1)
    heatmap(time_series, colorscale="rdpu")
    return ca, time_series


@app.cell
def _(StochasticLenia, heatmap):
    lenia = StochasticLenia(shape=(40, 40))
    ts = lenia.time_steps(200)
    heatmap(ts, colorscale="viridis")
    return lenia, ts


@app.cell
def _(MCSTLattice, heatmap):
    mcst = MCSTLattice(shape=(40, 40))
    ts_ = mcst.time_steps(200)
    heatmap(ts_, title="MCST heatmap", colorscale="redor")
    return mcst, ts_


if __name__ == "__main__":
    app.run()
