import marimo

__generated_with = "0.11.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    from src.lattice.ca import Rule224
    from src.utils.plot import discrete_heatmap
    from src.models.embeddings import RopeEmbedding, EmbeddingWrapper, StateEmbedding, ComposeEmbeddings
    return (
        ComposeEmbeddings,
        EmbeddingWrapper,
        RopeEmbedding,
        Rule224,
        StateEmbedding,
        discrete_heatmap,
        mo,
        nn,
        torch,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ### 2D-RoPE positional embeddings

        flatten to sequence -> apply RoPE to positions
        """
    )
    return


@app.cell
def _(
    ComposeEmbeddings,
    EmbeddingWrapper,
    RopeEmbedding,
    StateEmbedding,
    torch,
):
    batch_size = 2
    dim = 100
    input_dim = 3
    d_model = 64

    lattice_data = torch.randn(batch_size, dim, dim, input_dim)

    state_embed = StateEmbedding(input_dim, d_model)
    pos_embed = EmbeddingWrapper(RopeEmbedding, d_model=d_model, max_dim=dim, base=1_000)

    embedding = ComposeEmbeddings([state_embed, pos_embed], combine_fn="sum")

    final_embedding = embedding(lattice_data)
    print("Final embedding shape:", final_embedding.shape)
    return (
        batch_size,
        d_model,
        dim,
        embedding,
        final_embedding,
        input_dim,
        lattice_data,
        pos_embed,
        state_embed,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
