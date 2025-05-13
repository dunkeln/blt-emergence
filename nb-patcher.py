

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
    from torch.utils.data import DataLoader
    from einops import rearrange

    from stochasticlm.dynamics import Rule224, CTMCActiveLattice
    from stochasticlm.utils import heatmap, LatticeSeqDataset, learn_lm
    from stochasticlm.utils.patchinglm import train, eval
    from stochasticlm.models import PatchingTransformer, Patcher
    from stochasticlm.utils.checks import verify_patch_counts

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    return (
        CTMCActiveLattice,
        DataLoader,
        LatticeSeqDataset,
        Patcher,
        Rule224,
        verify_patch_counts,
    )


@app.cell
def _(CTMCActiveLattice, DataLoader, LatticeSeqDataset, Rule224):
    gol_loader = DataLoader(LatticeSeqDataset(Rule224, shape=(20, 20)), batch_size=8, shuffle=False)
    ctmc_loader = DataLoader(LatticeSeqDataset(CTMCActiveLattice, shape=(20, 20), seq_length=64), batch_size=8, shuffle=False, pin_memory=True)
    return (ctmc_loader,)


@app.cell
def _(Patcher):
    gol_uri = 'runs:/245da404fa404266b019fd238df9004b/model'
    ctmc_uri = 'runs:/d2db75446a8f489db1aea1a00039b9cb/model'

    test_patcher = Patcher(model_uri=ctmc_uri, max_patch_len=15)
    return (test_patcher,)


@app.cell
def _(ctmc_loader, test_patcher):
    inp, tgt = next(iter(ctmc_loader))
    tokenized, mask = test_patcher.batch_patch(inp, kind="delta")
    return inp, tokenized


@app.cell
def _(inp, tokenized, verify_patch_counts):
    verify_patch_counts(tokenized, inp, pad_id=5)
    return


if __name__ == "__main__":
    app.run()
