from functools import reduce
import mlflow
import mlflow.pytorch as mlflowpytorch
import marimo as mo
import os
import psutil
import torch
import pynvml

def compose(*fns):
    def __inner__(*args):
        return reduce(lambda acc, fn: (fn(*acc) if isinstance(acc, tuple) else fn(acc)), reversed(fns), args)
    return __inner__


def learn_lm(config: dict):
    # INFO: load all params from config
    model = config['model']
    optimizer = config['optimizer']
    criterion = config['criterion']
    train_loader = config['train_loader']
    val_loader = config['val_loader']
    num_epochs = config.get('num_epochs', 2)
    train_fn = config['train_fn']
    eval_fn = config['eval_fn']
    kind = config.get('kind', 'GoL')
    experiment = config.get('experiment', 'default')
    device = config['device']
    tag = config.get('tag', 'test')
    version = config.get('version', '0.0.0')
    patcher_uri = config.get('patcher_uri')
    max_patch_len = config.get('max_patch_len')


    step_size = len(val_loader) + len(train_loader)

    gpu_handle = None
    if device.type == "cuda":
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    mlflow.end_run()
    mlflow.set_experiment(experiment)
    mlflow.set_experiment_tag(tag, version)


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
        mlflow.set_tag(key="dynamics", value=kind)

        with mo.status.progress_bar(
            total=num_epochs * step_size,
            title=f"{experiment} | {kind}",
            subtitle="training...",
            show_eta=True,
            show_rate=True,
            completion_subtitle=f"run id: {run.info.run_id}",
            completion_title=f"logged at {run.info.run_name}.",
            remove_on_exit=False
        ) as bar:
            for epoch in range(num_epochs):
                if patcher_uri and max_patch_len:
                    train_fn(model, patcher_uri, max_patch_len, train_loader, optimizer, criterion, device, epoch)
                else:
                    train_fn(model, train_loader, optimizer, criterion, device, epoch)

                bar.update(subtitle="evaluating...", increment=len(train_loader))

                if patcher_uri and max_patch_len:
                    eval_fn(model, patcher_uri, max_patch_len, val_loader, criterion, device, epoch)
                else:
                    eval_fn(model, val_loader, criterion, device, epoch)
                cpu_pct = psutil.cpu_percent()
                ram_pct = psutil.virtual_memory().percent
                rss_gb  = psutil.Process(os.getpid()).memory_info().rss / (1024**3)

                mlflow.log_metric("system/cpu_percent", cpu_pct, step=epoch)
                mlflow.log_metric("system/ram_percent", ram_pct, step=epoch)
                mlflow.log_metric("system/process_rss_gb", rss_gb,  step=epoch)

                # INFO: GPU (CUDA) or MPS (Apple Silicon)
                if gpu_handle and device.type == 'cuda':
                    util     = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
                    gpu_mem  = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)

                    util = float(util)
                    gpu_mem = float(gpu_mem.used) / (1024 ** 3)

                    mlflow.log_metric("system/gpu_util_percent", util, step=epoch)
                    mlflow.log_metric("system/gpu_mem_gb", gpu_mem, step=epoch)

                elif device.type == "mps":
                    # INFO: allocator stats from PyTorch MPS
                    reserved = torch.mps.current_allocated_memory()
                    allocated = torch.mps.driver_allocated_memory()
                    mlflow.log_metric("system/mps_reserved_mb",   reserved   / 1e6, step=epoch)
                    mlflow.log_metric("system/mps_allocated_mb",  allocated  / 1e6, step=epoch)

                bar.update(subtitle="training...", increment=len(val_loader))

        # INFO: logging model weights for the run
        mlflowpytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            code_paths=["./stochasticlm/models/patching_lm.py"],
        )
