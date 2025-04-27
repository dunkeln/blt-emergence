

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch
    from torch.utils.data import DataLoader

    from src.lattice import Rule224, StochasticLenia, MCSTLattice
    from src.utils import heatmap, LatticeSeqDataset
    return (
        DataLoader,
        LatticeSeqDataset,
        MCSTLattice,
        Rule224,
        StochasticLenia,
        heatmap,
        mo,
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
    return


@app.cell
def _(StochasticLenia, heatmap):
    lenia = StochasticLenia(shape=(40, 40))
    ts = lenia.time_steps(200)
    heatmap(ts, colorscale="viridis")
    return


@app.cell
def _(MCSTLattice, heatmap):
    mcst = MCSTLattice(shape=(20, 20))
    ts_ = mcst.time_steps(200)
    heatmap(ts_, title="MCST heatmap", colorscale="redor")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### DataLoader for LM training

        > subsection demonstrates working with a DataLoader for the simulations

        + Input *carries sequence of inputs*
        + Target *carries next step of the input sequences*
        """
    )
    return


@app.cell
def _(DataLoader, LatticeSeqDataset, Rule224):
    ds = LatticeSeqDataset(Rule224)
    loader = DataLoader(ds, batch_size=8, shuffle=True)

    for batch_idx, (inp, tgt) in enumerate(loader):
        # inp, tgt are each of shape (batch, seq_len, H, W)
        print(f"Batch {batch_idx}:")
        print("  input  :", inp.shape)   # → torch.Size([8, 32, 20, 20])
        print("  target :", tgt.shape)   # → torch.Size([8, 32, 20, 20])
        break
    return (inp,)


@app.cell
def _(heatmap, inp):
    heatmap(inp[0,...])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
