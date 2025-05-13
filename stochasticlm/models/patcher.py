import mlflow
import torch
import torch.nn.functional as F

# WARN: have to keep a pad_id with an integer value to declare loss fn instance before introducing the masked patches
class Patcher:
    def __init__(self, model_uri: str, threshold: float = 0, max_patch_len: int = 10, kind="delta", pad_id=-1000):
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.threshold = threshold
        self.max_patch_len = max_patch_len
        self.kind = kind
        self.pad_id = pad_id

    def patch(self, lattice, kind=None):
        if kind:
            self.kind = kind

        if lattice.dim() > 1:
            lattice = lattice.flatten()

        N = lattice.numel()
        if self.pad_id is -1000:
            self.pad_id = int(lattice.max().item()) + 1
        logits = self.model.predict(
            lattice.unsqueeze(0).unsqueeze(0).numpy()
        )

        logits = torch.from_numpy(logits)
        probs = F.softmax(logits[0], dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        diffs = entropy[1:] - entropy[:-1]
        cuts = [0]
        match self.kind:
            case "delta":
                delta_idxs = (diffs > 0.1).nonzero(as_tuple=False).flatten().add(1)
                cuts = [0] + delta_idxs.tolist() + [N]

            case "percentile":
                pct  = torch.quantile(entropy, .95)
                cuts = [0] + [i for i, h in enumerate(entropy.tolist(), start=1) if h > pct] + [N]

            case "combined" | _:
                cond1 = entropy > self.threshold
                cond2 = torch.cat([torch.tensor([False], device='cpu'), diffs > 0.1])
                combined_idxs = (cond1 | cond2).nonzero(as_tuple=False).flatten().tolist()
                cuts = [0] + combined_idxs + [N]

        cuts = sorted(set(cuts))

        patches = []
        for start, end in zip(cuts[:-1], cuts[1:]):
            segment = lattice[start:end]
            length = segment.numel()
            if length == 0:
                continue
            for i in range(0, length, self.max_patch_len):
                chunk = segment[i:i + self.max_patch_len]
                if chunk.numel() < self.max_patch_len:
                    chunk = F.pad(
                        chunk,
                        (0, self.max_patch_len - chunk.numel()),
                        value=self.pad_id
                    )
                patches.append(chunk)

        patches = torch.stack(patches)
        return entropy, patches

    def batch_patch(self, lattice_seq, kind="delta"):
        B, T, _, _ = lattice_seq.size()
        batch_patches = []
        pad_id = int(lattice_seq.max().item()) + 1

        for b in range(B):
            patches_b = []
            for t in range(T):
                _, patch = self.patch(lattice_seq[b, t], kind)
                patches_b.append(patch)
            batch_patches.append(patches_b)

        # INFO: more uniform patches
        patches_counts = [
            [p.size(0) for p in seq_patches]
            for seq_patches in batch_patches
        ]

        max_patches = max(max(row) for row in patches_counts)

        all_patches = torch.full((B, T, max_patches, self.max_patch_len), pad_id, dtype=torch.long, device='cpu')
        mask = torch.zeros((B, T, max_patches), dtype=torch.bool, device='cpu')

        for b in range(B):
            for t in range(T):
                patches_bt = batch_patches[b][t]
                m_bt = patches_bt.size(0)
                all_patches[b, t, :m_bt] = patches_bt
                mask[b, t, :m_bt] = True

        return all_patches, mask
