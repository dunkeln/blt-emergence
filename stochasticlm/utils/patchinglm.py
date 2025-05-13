import mlflow
import mlflow.pytorch
import torch
import math


def train(model, dataloader, optimizer, criterion, device, epoch=0, log_interval=25):
    """
    Training loop with MLflow logging:
      - per-iteration metrics: 'itvl/train/loss', 'itvl/train/accuracy'
      - per-log_interval averages: 'batch/train/avg_loss', 'batch/train/avg_accuracy'
      - per-epoch metrics: 'epoch/train/loss', 'epoch/train/accuracy', 'epoch/train/bpc'
    """
    model.train()
    total_loss = total_tokens = total_correct = 0
    running_loss = running_correct = 0
    global_step = epoch * len(dataloader)

    for idx, (inp, tgt) in enumerate(dataloader):
        inp, tgt = inp.to(device), tgt.to(device)
        B, T, H, W = inp.shape
        x = inp.view(B * T, H, W)
        y = tgt.view(B * T, H, W)

        logits = model(x)
        V = logits.size(-1)

        loss = criterion(logits.view(-1, V), y.view(-1))
        preds = logits.argmax(dim=-1)
        y_flat = y.view(B * T, H * W)
        batch_correct = (preds == y_flat).sum().item()
        batch_tokens = B * T * H * W

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        total_loss   += loss.item() * batch_tokens
        total_tokens += batch_tokens
        total_correct+= batch_correct


        running_loss   += loss.item() * batch_tokens
        running_correct+= batch_correct
        global_step   += 1



        mlflow.log_metric('itvl/train/loss', loss.item(), step=global_step)
        mlflow.log_metric('itvl/train/accuracy', batch_correct / batch_tokens, step=global_step)


        if (idx + 1) % log_interval == 0:
            avg_loss = running_loss / (log_interval * batch_tokens)
            avg_acc  = running_correct / (log_interval * batch_tokens)
            mlflow.log_metric('batch/train/avg_loss', avg_loss, step=global_step)
            mlflow.log_metric('batch/train/avg_accuracy', avg_acc, step=global_step)
            running_loss = running_correct = 0



    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    bpc      = avg_loss / math.log(2)

    mlflow.log_metric('epoch/train/loss', avg_loss, step=epoch)
    mlflow.log_metric('epoch/train/accuracy', accuracy, step=epoch)
    mlflow.log_metric('epoch/train/bpc', bpc, step=epoch)

    return avg_loss, accuracy, bpc


def eval(model, dataloader, criterion, device, epoch=0, log_interval=25):
    """
    Validation loop with MLflow logging:
      - per-iteration metrics: 'itvl/val/loss', 'itvl/val/accuracy'
      - per-epoch metrics:    'epoch/val/loss', 'epoch/val/accuracy', 'epoch/val/bpc'
    """
    model.eval()
    total_loss = total_tokens = total_correct = 0
    running_loss = running_correct = 0
    global_step = epoch * len(dataloader)

    with torch.no_grad():
        for idx, (inp, tgt) in enumerate(dataloader):
            inp, tgt = inp.to(device), tgt.to(device)
            B, T, H, W = inp.shape
            x = inp.view(B * T, H, W)
            y = tgt.view(B * T, H, W)

            logits = model(x)
            V = logits.size(-1)

            loss = criterion(logits.view(-1, V), y.view(-1))
            preds = logits.argmax(dim=-1)
            y_flat = y.view(B * T, H * W)
            batch_correct = (preds == y_flat).sum().item()
            batch_tokens  = B * T * H * W


            total_loss   += loss.item() * batch_tokens
            total_tokens += batch_tokens
            total_correct+= batch_correct


            if log_interval:
                running_loss   += loss.item() * batch_tokens
                running_correct+= batch_correct

            global_step += 1

            mlflow.log_metric('itvl/val/loss', loss.item(), step=global_step)
            mlflow.log_metric('itvl/val/accuracy', batch_correct / batch_tokens, step=global_step)


            if log_interval and (idx + 1) % log_interval == 0:
                avg_loss = running_loss / (log_interval * batch_tokens)
                avg_acc  = running_correct / (log_interval * batch_tokens)
                mlflow.log_metric('batch/val/avg_loss', avg_loss, step=global_step)
                mlflow.log_metric('batch/val/avg_accuracy', avg_acc, step=global_step)
                running_loss = running_correct = 0


    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    bpc      = avg_loss / math.log(2)

    mlflow.log_metric('epoch/val/loss', avg_loss, step=epoch)
    mlflow.log_metric('epoch/val/accuracy', accuracy, step=epoch)
    mlflow.log_metric('epoch/val/bpc', bpc, step=epoch)

    return avg_loss, accuracy, bpc
