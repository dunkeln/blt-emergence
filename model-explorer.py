import marimo

__generated_with = "0.11.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn.functional as F
    from src.lattice.ca import BaseModel
    from src.utils import discrete_heatmap
    return BaseModel, F, discrete_heatmap, mo, torch


@app.cell
def _(mo):
    mo.md(
        r"""
        # Exploratory phase of BLT architecture

        **Problems:**   
        Traditional tokenization too strict.   
        Byte-Latent Transformer encoding prevails in 1-D.   
        ViT encodes strict patches.
        """
    )
    return


@app.cell
def _(BaseModel, discrete_heatmap):
    ca = BaseModel(shape=(15, 15))
    discrete_heatmap(ca.get_batches())
    return (ca,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Goal

        + foresee next state
        + multi-step forecasting   
            *o/p should have same encoding as input*
        """
    )
    return


if __name__ == "__main__":
    app.run()
