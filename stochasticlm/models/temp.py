import mlflow
import torch
import torch.nn.functional as F
from einops import rearrange

# BUG: potential model prediction issue for Large LM due to return dtype
# INFO:
# trained model is retrieved from mlflow artifacts
class Patcher:
    def __init__(self, model_uri: str, threshold: float, max_patch_len: int, device: str="cpu"):
        # Load MLflow pyfunc model for patching
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.threshold = threshold
        self.max_len = max_patch_len
        self.device = torch.device(device)

    def _tokenize_single(self, flat_tensor):
        """
        Tokenize a single flattened lattice (1D tensor of length N).
        Returns a tensor of shape (M, max_patch_len).
        """
        N = flat_tensor.numel()
        # Run through pyfunc to get logits
        batch_flat = flat_tensor.unsqueeze(0).cpu().numpy()      # (1, N)
        logits = self.model.predict(batch_flat)                 # numpy (1, N, V)
        # Compute entropy per position
        probs = torch.from_numpy(logits[0]).softmax(dim=-1)     # (N, V)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)  # (N,)

        cutoffs = [0]
        for i, h in enumerate(entropy.tolist(), start=1):
            if h < self.threshold:
                cutoffs.append(i)
        cutoffs.append(N)

        patches = []
        for start, end in zip(cutoffs, cutoffs[1:]):
            patch = flat_tensor[start:end]
            L = patch.numel()
            if L > self.max_len:
                patch = patch[:self.max_len]
            else:
                patch = F.pad(patch, (0, self.max_len - L), value=0)
            patches.append(patch)

        return torch.stack(patches, dim=0)  # (M, max_len)

    def tokenize(self, lattice):
        """
        Tokenize input lattices into patches. Accepts:
          - 3D tensor (B', H, W): batch of 2D lattices flattened over time.
          - 2D tensor (H, W): single lattice.
          - 1D tensor (N,): single flattened lattice.

        Returns a list of patch tensors per example, each of shape (M_i, max_patch_len).
        """
        # Case: batch of lattices
        if lattice.dim() == 3:
            Bp, H, W = lattice.shape
            results = []
            for i in range(Bp):
                grid = lattice[i]
                flat = rearrange(grid, 'h w -> (h w)').to(self.device)
                results.append(self._tokenize_single(flat))
            return results

        # Case: single 2D lattice
        if lattice.dim() == 2:
            flat = rearrange(lattice, 'h w -> (h w)').to(self.device)
            return self._tokenize_single(flat)

        # Case: single flattened lattice
        if lattice.dim() == 1:
            return self._tokenize_single(lattice.to(self.device))

        raise ValueError(f"Unsupported tensor shape: {tuple(lattice.shape)}")
