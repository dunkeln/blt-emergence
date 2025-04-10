import marimo

__generated_with = "0.11.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    from src.lattice import Rule224, MCSTLattice
    from src.utils.plot import heatmap
    # from src.model.embeddings import RopeEmbedding, EmbeddingWrapper, StateEmbedding, ComposeEmbeddings
    from src.model.patching_lm import AutoregressiveLM
    return AutoregressiveLM, MCSTLattice, Rule224, heatmap, mo, nn, torch


@app.cell
def _(mo):
    mo.md(
        r"""
        ### TODOS:
        - [x] 2D-RoPE positional embeddings
        - [ ] convolutional embeddings
        """
    )
    return


@app.cell
def _(MCSTLattice, Rule224):
    mcst = MCSTLattice(shape=(40, 40))
    gol = Rule224(shape=(40, 40))
    return gol, mcst


@app.cell
def _(mcst):
    mcst.size(0)
    return


@app.cell
def _(AutoregressiveLM, torch):
    B = 2       # batch size
    H = 10      # height
    W = 10      # width
    vocab_size = 256
    d_model = 64
    nhead = 4
    num_layers = 2

    model = AutoregressiveLM(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        height=H,
        width=W
    )

    dummy_input = torch.randint(0, vocab_size, (B, 1, H, W))  # shape (2, 1, 10, 10)
    output_logits = model(dummy_input)
    print("Output shape:", output_logits.shape)
    # Should be (B, H*W, vocab_size) => (2, 100, 256) in this example.
    return (
        B,
        H,
        W,
        d_model,
        dummy_input,
        model,
        nhead,
        num_layers,
        output_logits,
        vocab_size,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
