import marimo

__generated_with = "0.11.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import plotly.graph_objects as go

    from src.lattice.ca import BaseModel
    from src.lattice.ca.rule_224 import Rule_224
    return BaseModel, Rule_224, go, mo, np, pl


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
def _(Rule_224):
    ca = Rule_224()
    return (ca,)


@app.cell
def _(ca):
    ca.show()
    return


@app.cell
def _(ca, np):
    time_series = np.stack([next(ca).lattice for _ in range(10)], axis=0)
    time_series = np.repeat(time_series, 5, axis=0)
    return (time_series,)


@app.cell
def _(go, np, time_series):
    x, y, z = np.where(time_series == 1)

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=3, color=z, colorscale='Viridis', opacity=0.8)
        )
    ])

    # Set axis labels and display
    fig.update_layout(
        title="3D Cellular Automata Lattice",
        scene=dict(
            xaxis_title="time ->",
            yaxis_title="x Axis",
            zaxis_title="Y Axis"
        )
    )

    fig.show()
    return fig, x, y, z


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
