

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import mlflow
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from einops import rearrange

    from stochasticlm.dynamics import Rule224, CTMCActiveLattice
    from stochasticlm.utils import heatmap, LatticeSeqDataset, learn_lm
    from stochasticlm.utils.largelm import train, eval, to_lattice_logits
    from stochasticlm.models import PatchingTransformer, Patcher, LargeLM
    from stochasticlm.utils.checks import verify_patch_counts

    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    return (
        CTMCActiveLattice,
        DataLoader,
        LargeLM,
        LatticeSeqDataset,
        Patcher,
        Rule224,
        eval,
        learn_lm,
        mo,
        nn,
        torch,
        train,
    )


@app.cell
def _(CTMCActiveLattice, DataLoader, LatticeSeqDataset, Rule224):
    gol_loader = DataLoader(LatticeSeqDataset(Rule224, shape=(20, 20)), batch_size=8, shuffle=False)
    ctmc_loader = DataLoader(LatticeSeqDataset(CTMCActiveLattice, shape=(20, 20), seq_length=64), batch_size=8, shuffle=False, pin_memory=True)
    return


@app.cell
def _(Patcher):
    gol_uri = 'runs:/245da404fa404266b019fd238df9004b/model'
    ctmc_uri = 'runs:/d2db75446a8f489db1aea1a00039b9cb/model'

    test_patcher = Patcher(model_uri=ctmc_uri, max_patch_len=15)
    return (gol_uri,)


@app.cell
def _(mo):
    train_btn = mo.ui.run_button(label="train model", kind="warn")
    train_btn
    return (train_btn,)


@app.cell
def _(
    DataLoader,
    LargeLM,
    LatticeSeqDataset,
    Rule224,
    eval,
    gol_uri,
    learn_lm,
    mo,
    nn,
    torch,
    train,
    train_btn,
):
    mo.stop(not train_btn.value)

    ds = LatticeSeqDataset(Rule224, shape=(20, 20))
    model = LargeLM(num_states=2, d_model=128, enc_heads=4,latent_layers=6,latent_heads=8,max_patch_len=15)
    config = {
        'experiment': 'global_lm',
        'tag': 'test',
        'tag_version': '0.0.0',
        'kind': 'GoL',
        'num_epochs': 2,
        'model': model,
        'device': torch.device('mps'),
        'optimizer': torch.optim.Adam(model.parameters(), lr=1e-4),
        'criterion': nn.CrossEntropyLoss(),
        'train_loader': DataLoader(ds, batch_size=32, shuffle=True),
        'val_loader': DataLoader(ds, batch_size=32, shuffle=False),
        'train_fn': train,
        'eval_fn': eval,
        'patcher_uri': gol_uri,
        'max_patch_len': 15
    }

    learn_lm(config)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
