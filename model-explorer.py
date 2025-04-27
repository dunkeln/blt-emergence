

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
        eval,
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
    ds = LatticeSeqDataset(Rule224, shape=(50, 50))
    train_loader = DataLoader(ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(ds, batch_size=8, shuffle=False)
    return train_loader, val_loader


@app.cell
def _(eval, model, nn, torch, train, train_loader, val_loader):
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 2

    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval(model, val_loader, criterion, device)
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    return criterion, optimizer


@app.cell
def _(torch):
    def train_epoch(model, dataloader, optimizer, criterion, device):
        """
        Train the model for one epoch to predict next‐frame lattice states.
        - model: PatchingTransformer
        - dataloader: yields (inp, tgt) of shape (B, seq_len, H, W)
        - criterion: nn.CrossEntropyLoss()
        """
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for inp, tgt in dataloader:
            # inp, tgt: (B, T, H, W)
            inp = inp.to(device)
            tgt = tgt.to(device)
            B, T, H, W = inp.shape

            # merge time and batch dims so each frame is a separate example
            x = inp.view(B * T, H, W)   # (B*T, H, W)
            y = tgt.view(B * T, H, W)   # (B*T, H, W)

            logits = model(x)           # (B*T, H*W, V)
            V = logits.size(-1)

            # compute cross‐entropy over all cells
            loss = criterion(
                logits.view(-1, V),     # (B*T*H*W, V)
                y.view(-1)              # (B*T*H*W,)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate
            n_tokens = B * T * H * W
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

        return total_loss / total_tokens


    def eval_epoch(model, dataloader, criterion, device):
        """
        Evaluate the model on validation set, returning avg loss and accuracy.
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        with torch.no_grad():
            for inp, tgt in dataloader:
                inp = inp.to(device)
                tgt = tgt.to(device)
                B, T, H, W = inp.shape

                x = inp.view(B * T, H, W)
                y = tgt.view(B * T, H, W)

                logits = model(x)       # (B*T, H*W, V)
                V = logits.size(-1)

                loss = criterion(
                    logits.view(-1, V),
                    y.view(-1)
                )

                preds = logits.argmax(dim=-1)  # (B*T, H*W)
                total_correct += (preds == y.view(B * T, H * W)).sum().item()

                n_tokens = B * T * H * W
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens

        avg_loss = total_loss / total_tokens
        accuracy = total_correct / total_tokens
        return avg_loss, accuracy
    return (train_epoch,)


@app.cell
def _(criterion, model, optimizer, train_epoch, train_loader):
    train_epoch(model, train_loader, optimizer, criterion, device="mps")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
