import mlflow
import torch
from .patching_lm import PatchingTransformer

class PatcherTokenizer:
    def __init__(self,
                 run_id: str,
                 artifact_path: str,
                 threshold: float,
                 num_states: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 device: str = 'cpu'):

        # INFO: Download the .pth artifact
        local_path = mlflow.artifacts.download_artifacts(artifact_path=artifact_path, run_id=run_id)
        ckpt = torch.load(local_path, map_location=device)
        # instantiate patching LM and load
        self.model = PatchingTransformer(num_states, d_model,
                                         num_layers, num_heads)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(device).eval()

        self.threshold = threshold
        self.device = device

    def tokenize(self, lattice: torch.LongTensor):
        """
        lattice: (H, W) LongTensor with values in [0..num_states-1]
        returns: list of 1D Tensors (patch sequences)
        """
        H, W = lattice.shape
        flat = lattice.view(-1).unsqueeze(0).to(self.device)  # (1, H*W)
        with torch.no_grad():
            logits = self.model(flat)                           # (1, H*W, V)
            ent = self.model.compute_entropy(logits).squeeze(0)  # (H*W,)
        ent_list = ent.cpu().tolist()
        N = len(ent_list)
        # compute cut positions
        cuts = [0] + [i for i, h in enumerate(ent_list, 1) if h > self.threshold] + [N]
        patches = []
        data = lattice.view(-1)
        for i in range(len(cuts)-1):
            patches.append(data[cuts[i]:cuts[i+1]])
        return patches
