

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
    from src.lattice import Rule224, MCSTLattice
    from src.utils import heatmap, LatticeSeqDataset, train, eval
    import plotly.graph_objects as go
    from src.model import RoPE, EncoderBlock, PatchingTransformer
    return (
        DataLoader,
        LatticeSeqDataset,
        MCSTLattice,
        PatchingTransformer,
        Rule224,
        heatmap,
        mo,
        nn,
        torch,
        train,
    )


@app.cell
def _(MCSTLattice, torch):
    ca = MCSTLattice(shape=(50, 50), dtype=torch.int8)
    time_series = ca.time_steps(100)
    return ca, time_series


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
    print(x.size())
    model = PatchingTransformer(num_embeddings=5)
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
def _(ca):
    ca.lattice.unique()
    return


@app.cell
def _(logits, model):
    entropies = model.compute_entropy(logits)
    return (entropies,)


@app.cell
def _(entropies, heatmap):
    heatmap(entropies.reshape(-1, 50, 50).detach().numpy())
    return


@app.cell
def _(mo):
    mo.md(r"""### Training Loop""")
    return


@app.cell
def _(DataLoader, LatticeSeqDataset, Rule224):
    ds = LatticeSeqDataset(Rule224, shape=(20, 20))
    train_loader = DataLoader(ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(ds, batch_size=8, shuffle=False)
    return (train_loader,)


@app.cell
def _(model, nn, torch):
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 2
    return criterion, device, optimizer


@app.cell
def _(criterion, device, model, optimizer, train, train_loader):
    train(model, train_loader, optimizer, criterion, device=device)
    return


@app.cell
async def _(mo):
    import asyncio

    for _ in mo.status.progress_bar(
        range(10),
        title="training...",
        subtitle="Please wait",
        show_eta=True,
        show_rate=True
    ):
        await asyncio.sleep(0.5)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
