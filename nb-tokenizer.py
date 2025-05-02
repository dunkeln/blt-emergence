

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
    import plotly.graph_objects as go
    import mlflow
    import mlflow.pytorch
    import os, psutil

    try:
        import pynvml
    except ImportError:
        pynvml = None

    from stochasticlm.dynamics import Rule224, MCSTLattice
    from stochasticlm.utils import heatmap, LatticeSeqDataset
    from stochasticlm.utils.patchinglm import train, eval
    from stochasticlm.models import PatchingTransformer

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    return (
        DataLoader,
        LatticeSeqDataset,
        PatchingTransformer,
        Rule224,
        eval,
        mlflow,
        mo,
        nn,
        os,
        psutil,
        pynvml,
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
def _(DataLoader, LatticeSeqDataset, Rule224):
    ds = LatticeSeqDataset(Rule224, shape=(20, 20))
    train_loader = DataLoader(ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(ds, batch_size=8, shuffle=False)
    return train_loader, val_loader


@app.cell
def _(PatchingTransformer, nn, pynvml, torch):
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

    use_cuda = device.type == "cuda"
    if use_cuda:
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    model = PatchingTransformer(d_model=8, num_embeddings=2)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 2
    return (
        criterion,
        device,
        gpu_handle,
        model,
        num_epochs,
        optimizer,
        use_cuda,
    )


@app.cell
def _(mo, train_loader, val_loader):
    step_size = len(val_loader) + len(train_loader)
    train_btn = mo.ui.run_button(label="train model", kind="warn")
    train_btn
    return step_size, train_btn


@app.cell
def _(
    criterion,
    device,
    eval,
    gpu_handle,
    mlflow,
    mo,
    model,
    num_epochs,
    optimizer,
    os,
    psutil,
    pynvml,
    step_size,
    torch,
    train,
    train_btn,
    train_loader,
    use_cuda,
    val_loader,
):
    mo.stop(not train_btn.value)

    mlflow.end_run()
    mlflow.set_experiment("patching_lm")
    mlflow.set_experiment_tag("test:GoL", "0.0.0")

    with mlflow.start_run() as run:
        mlflow.log_params({
            "optimizer": optimizer.__class__.__name__,
            "loss_fn": criterion.__class__.__name__,
            "epochs": num_epochs,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "d_model": model.d_model,
            "num_embeddings": model.num_embeddings,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
        })
        mlflow.set_tag(key="dynamics", value="GoL")

        with mo.status.progress_bar(
            total=num_epochs * step_size,
            title="Patching LM",
            subtitle="training...",
            show_eta=True,
            show_rate=True,
            completion_subtitle=f"run id: {run.info.run_id}",
            completion_title=f"logged at {run.info.run_name}.",
            remove_on_exit=False
        ) as bar:
            for epoch in range(num_epochs):
                avg_loss, accuracy, bpc = train(model, train_loader, optimizer, criterion, device, epoch)
                bar.update(subtitle="evaluating...", increment=len(train_loader))
                avg_loss, accuracy, bpc = eval(model, val_loader, criterion, device, epoch)
                cpu_pct = psutil.cpu_percent()
                ram_pct = psutil.virtual_memory().percent
                rss_gb  = psutil.Process(os.getpid()).memory_info().rss / (1024**3)

                mlflow.log_metric("system/cpu_percent",     cpu_pct, step=epoch)
                mlflow.log_metric("system/ram_percent",     ram_pct, step=epoch)
                mlflow.log_metric("system/process_rss_gb",  rss_gb,  step=epoch)

                # INFO: GPU (CUDA) or MPS (Apple Silicon)
                if use_cuda:
                    util     = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
                    gpu_mem  = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / (1024**3)
                    mlflow.log_metric("system/gpu_util_percent", util,      step=epoch)
                    mlflow.log_metric("system/gpu_mem_gb",       gpu_mem,   step=epoch)

                elif device.type == "mps":
                    # INFO: allocator stats from PyTorch MPS
                    reserved = torch.mps.current_allocated_memory()
                    allocated = torch.mps.driver_allocated_memory()
                    mlflow.log_metric("system/mps_reserved_mb",   reserved   / 1e6, step=epoch)
                    mlflow.log_metric("system/mps_allocated_mb",  allocated  / 1e6, step=epoch)

                bar.update(subtitle="training...", increment=len(val_loader))

        # INFO: logging model weights for the run
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            code_paths=["./stochasticlm/models/patching_lm.py"],
        )
    return


if __name__ == "__main__":
    app.run()
