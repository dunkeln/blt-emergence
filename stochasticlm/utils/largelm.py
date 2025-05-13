import math
from pandas._config.config import re
import torch
import torch.nn as nn
from einops import rearrange
import mlflow

from stochasticlm.models.patcher import Patcher

def train(model, patcher_uri, max_patch_len, dataloader,
          optimizer, criterion, device, epoch=0, log_interval=25):
    model.train()
    total_loss = total_tokens = total_correct = 0
    running_loss = running_tokens = running_correct = 0
    global_step = epoch * len(dataloader)


    sample_lattice, _ = next(iter(dataloader))
    pad_id = int(sample_lattice.max().item()) + 1
    patcher = Patcher(model_uri=patcher_uri,
                      max_patch_len=max_patch_len,
                      pad_id=pad_id)



    for idx, (inp, tgt) in enumerate(dataloader):
        B, T, H, W = inp.shape


        X_patches, patch_mask = patcher.batch_patch(inp, kind=patcher.kind)
        X_patches, patch_mask = X_patches.to(device), patch_mask.to(device)
        inp, tgt = inp.to(device), tgt.to(device)


        Bp, Tp, M, L = X_patches.shape
        logits = model(X_patches, patch_mask)


        lattice_logits = to_lattice_logits(
            logits, X_patches, patch_mask, (B, T, H, W)
        )


        ls = lattice_logits.permute(0, 1, 4, 2, 3)
        ls = rearrange(ls, 'b t v h w -> (b t) v h w')
        tg = rearrange(tgt, 'b t h w -> (b t) h w')
        ls = ls.to(device)


        loss = criterion(ls, tg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        n_cells = B * T * H * W
        n_correct = (ls.argmax(dim=1) == tg).sum().item()

        total_tokens   += n_cells
        total_loss     += loss.item() * n_cells
        total_correct  += n_correct

        running_tokens += n_cells
        running_loss   += loss.item() * n_cells
        running_correct+= n_correct

        global_step += 1
        mlflow.log_metric("step/train/loss",     loss.item(),           step=global_step)
        mlflow.log_metric("step/train/accuracy", n_correct / n_cells,  step=global_step)

        if idx % log_interval == 0 and running_tokens > 0:
            mlflow.log_metric("batch/train/loss",     running_loss / running_tokens,   step=global_step)
            mlflow.log_metric("batch/train/accuracy", running_correct / running_tokens, step=global_step)
            running_loss = running_tokens = running_correct = 0


    epoch_loss = total_loss / total_tokens
    epoch_acc  = total_correct / total_tokens
    epoch_bpc  = epoch_loss / math.log(2)

    mlflow.log_metric("epoch/train/loss",     epoch_loss, step=epoch)
    mlflow.log_metric("epoch/train/accuracy", epoch_acc,  step=epoch)
    mlflow.log_metric("epoch/train/bpc",      epoch_bpc,  step=epoch)

    return epoch_loss, epoch_acc, epoch_bpc


def eval(model, patcher_uri, max_patch_len, dataloader, criterion, device, epoch=0):
    model.eval()
    total_loss = total_tokens = total_correct = 0


    sample_lattice, _ = next(iter(dataloader))
    pad_id = int(sample_lattice.max().item()) + 1
    patcher = Patcher(model_uri=patcher_uri,
                      max_patch_len=max_patch_len,
                      pad_id=pad_id)



    with torch.no_grad():
        for idx, (inp, tgt) in enumerate(dataloader):
            B, T, H, W = inp.shape


            X_patches, patch_mask = patcher.batch_patch(inp, kind=patcher.kind)
            X_patches, patch_mask = X_patches.to(device), patch_mask.to(device)
            inp, tgt = inp.to(device), tgt.to(device)


            Bp, Tp, M, L = X_patches.shape
            logits = model(X_patches, patch_mask)


            lattice_logits = to_lattice_logits(
                logits, X_patches, patch_mask, (B, T, H, W)
            )


            ls = lattice_logits.permute(0, 1, 4, 2, 3)
            ls = rearrange(ls, 'b t v h w -> (b t) v h w')
            tg = rearrange(tgt, 'b t h w -> (b t) h w')
            ls = ls.to(device)


            loss = criterion(ls, tg)


            n_cells   = B * T * H * W
            n_correct = (ls.argmax(dim=1) == tg).sum().item()

            total_tokens  += n_cells
            total_loss    += loss.item() * n_cells
            total_correct += n_correct

            step = epoch * len(dataloader) + idx + 1
            mlflow.log_metric("step/val/loss",     loss.item(),          step=step)
            mlflow.log_metric("step/val/accuracy", n_correct / n_cells, step=step)

    avg_loss = total_loss / total_tokens
    avg_acc  = total_correct / total_tokens
    avg_bpc  = avg_loss / math.log(2)

    mlflow.log_metric("epoch/val/loss",     avg_loss, step=epoch)
    mlflow.log_metric("epoch/val/accuracy", avg_acc,  step=epoch)
    mlflow.log_metric("epoch/val/bpc",      avg_bpc,  step=epoch)

    return avg_loss, avg_acc, avg_bpc

def reconstruct_lattice(logits, patches, mask, size):
    preds = torch.argmax(logits, dim=-1)
    pm = rearrange(mask, 'b t p -> b t p 1')
    pm = pm.expand(-1, -1, -1, patches.size(-1))

    pad_id = torch.max(patches)
    cm = (patches != pad_id)
    keep = cm & pm
    keep_flat = rearrange(keep, 'b t p l -> (b t) (p l)')
    pred_flat = rearrange(preds, 'b t p l -> (b t) (p l)')
    kept = pred_flat.masked_select(keep_flat)
    return kept.view(*size)


def to_lattice_logits(logits, patches, patch_mask, size: tuple[int, int, int, int]) -> torch.Tensor:
    B, T, _, L, V = logits.shape
    B2, T2, H, W = size
    assert B == B2 and T == T2, "Batch/time mismatch"
    device = logits.device
    patches = patches.to(device)

    pm = rearrange(patch_mask, 'b t p -> b t p 1').expand(-1, -1, -1, L).to(device)

    pad_id = patches.max().item()
    cm = (patches != pad_id)

    keep = cm & pm

    logits_flat = rearrange(logits, 'b t p l v -> (b t p l) v')
    keep_flat   = rearrange(keep,    'b t p l -> (b t p l)').to(device)

    kept_logits = logits_flat[keep_flat]

    return kept_logits.view(B2, T2, H, W, V)
