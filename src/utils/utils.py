from functools import reduce
import torch

def compose(*fns):
    def __inner__(*args):
        return reduce(lambda acc, fn: (fn(*acc) if isinstance(acc, tuple) else fn(acc)), reversed(fns), args)
    return __inner__

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for inp, tgt in dataloader:
        inp = inp.to(device)
        tgt = tgt.to(device)
        B, T, H, W = inp.shape
        # INFO: merging time and batch dims so each frame is a separate example
        x = inp.view(B * T, H, W)
        y = tgt.view(B * T, H, W)

        logits = model(x)
        V = logits.size(-1)

        loss = criterion(
            logits.view(-1, V),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n_tokens = B * T * H * W
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    return total_loss / total_tokens


def eval(model, dataloader, criterion, device):
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

            logits = model(x)
            V = logits.size(-1)

            loss = criterion(
                logits.view(-1, V),
                y.view(-1)
            )

            preds = logits.argmax(dim=-1)
            total_correct += (preds == y.view(B * T, H * W)).sum().item()

            n_tokens = B * T * H * W
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    return avg_loss, accuracy
