

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from einops import rearrange
    import mlflow

    from stochasticlm.dynamics import Rule224, MCSTLattice
    from stochasticlm.utils import heatmap, LatticeSeqDataset, learn_lm
    from stochasticlm.utils.patchinglm import train, eval
    from stochasticlm.models import PatchingTransformer

    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    return (
        DataLoader,
        LatticeSeqDataset,
        MCSTLattice,
        PatchingTransformer,
        Rule224,
        device,
        eval,
        learn_lm,
        mo,
        nn,
        torch,
        train,
    )


@app.cell
def _(mo):
    mo.md("# AutoRegressive Patching LM").center()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### GoL Lattice

        training the Game of Life lattice on the patching LM and logging with `mlflow`.
        """
    )
    return


@app.cell
def _(mo):
    train_btn = mo.ui.run_button(label="train model", kind="warn")
    train_btn
    return (train_btn,)


@app.cell
def _(
    DataLoader,
    LatticeSeqDataset,
    PatchingTransformer,
    Rule224,
    device,
    eval,
    learn_lm,
    mo,
    nn,
    torch,
    train,
    train_btn,
):
    mo.stop(not train_btn.value)

    ds = LatticeSeqDataset(Rule224, shape=(20, 20))
    model = PatchingTransformer(d_model=8, num_embeddings=2).to(device)

    config = {
        'experiment': 'patching_lm',
        'tag': 'test',
        'tag_version': '0.0.2',
        'kind': 'GoL',
        'num_epochs': 2,
        'model': model,
        'device': device,
        'optimizer': torch.optim.Adam(model.parameters(), lr=1e-4),
        'criterion': nn.CrossEntropyLoss(),
        'train_loader': DataLoader(ds, batch_size=8, shuffle=True),
        'val_loader': DataLoader(ds, batch_size=8, shuffle=False),
        'train_fn': train,
        'eval_fn': eval,
    }

    learn_lm(config)
    return


@app.cell
def _(mo):
    train_btn_mcts = mo.ui.run_button(label="train model", kind="warn")
    train_btn_mcts
    return (train_btn_mcts,)


@app.cell
def _(
    DataLoader,
    LatticeSeqDataset,
    MCSTLattice,
    PatchingTransformer,
    device,
    eval,
    learn_lm,
    mo,
    nn,
    torch,
    train,
    train_btn_mcts,
):
    mo.stop(not train_btn_mcts.value)

    ds_mcts = LatticeSeqDataset(MCSTLattice, shape=(20, 20))
    model_mcts = PatchingTransformer(d_model=8, num_embeddings=5).to(device)

    config_mcts = {
        'experiment': 'patching_lm',
        'tag': 'test',
        'tag_version': '0.0.0',
        'kind': 'MCTS',
        'num_epochs': 2,
        'model': model_mcts,
        'device': device,
        'optimizer': torch.optim.Adam(model_mcts.parameters(), lr=1e-4),
        'criterion': nn.CrossEntropyLoss(),
        'train_loader': DataLoader(ds_mcts, batch_size=8, shuffle=True),
        'val_loader': DataLoader(ds_mcts, batch_size=8, shuffle=False),
        'train_fn': train,
        'eval_fn': eval,
    }

    learn_lm(config_mcts)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
