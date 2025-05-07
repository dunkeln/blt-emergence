

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch
    from torch.utils.data import DataLoader

    from stochasticlm.dynamics import Rule224, StochasticLenia, CTMCActiveLattice
    from stochasticlm.utils import heatmap, LatticeSeqDataset, seed_worker
    return (
        CTMCActiveLattice,
        DataLoader,
        LatticeSeqDataset,
        Rule224,
        StochasticLenia,
        heatmap,
        mo,
        seed_worker,
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
def _(CTMCActiveLattice, heatmap):
    ctmc = CTMCActiveLattice(shape=(20, 20))
    ts_ctmc = ctmc.time_steps(200)
    heatmap(ts_ctmc, title="CTMC Active Matter heatmap", colorscale="redor")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### DataLoader for LM training

        > subsection demonstrates working with a DataLoader for the simulations

        + Input *carries sequence of inputs*
        + Target *carries next step of the input sequences*

        In generating the random initial lattices, the dataloader is drawing i.i.d samples from the time series distribtion of the lattice transition.
        """
    )
    return


@app.cell
def _(CTMCActiveLattice, DataLoader, LatticeSeqDataset, seed_worker):
    ds = LatticeSeqDataset(CTMCActiveLattice)
    loader = DataLoader(ds, batch_size=8, shuffle=False, worker_init_fn=seed_worker, pin_memory=True)

    for batch_idx, (inp, tgt) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print("  input  :", inp.shape)
        print("  target :", tgt.shape)
        break
    return inp, tgt


@app.cell
def _(heatmap, inp):
    heatmap(inp[0,...])
    return


@app.cell
def _(heatmap, tgt):
    heatmap(tgt[0,...])
    return


@app.cell
def _(inp, torch):
    torch.unique(inp[0, 0, ...])
    return


if __name__ == "__main__":
    app.run()
