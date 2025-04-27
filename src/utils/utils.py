from functools import reduce
import torch

def compose(*fns):
    def __inner__(*args):
        return reduce(lambda acc, fn: (fn(*acc) if isinstance(acc, tuple) else fn(acc)), reversed(fns), args)
    return __inner__


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_tokens = 0.0, 0
    for inp, tgt in dataloader:
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer.zero_grad()
        print(f"pre-learning {inp.size()}")
        logits = model(inp)
        print("post-learning")
        loss   = criterion(
            logits.view(-1, logits.size(-1)), 
            tgt.view(-1)
        )
        loss.backward()
        optimizer.step()

        total_loss   += loss.item() * inp.numel()
        total_tokens += inp.numel()

    return total_loss / total_tokens

def eval(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_tokens, total_correct = 0.0, 0, 0
    with torch.no_grad():
        for inp, tgt in dataloader:
            inp, tgt = inp.to(device), tgt.to(device)
            logits = model(inp)
            _, _, N = logits.shape

            loss = criterion(
                logits.view(-1, N),
                tgt.view(-1)
            )
            total_loss   += loss.item() * inp.numel()
            total_tokens += inp.numel()

            preds = logits.argmax(dim=-1)
            total_correct += (preds == tgt).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    return avg_loss, accuracy
