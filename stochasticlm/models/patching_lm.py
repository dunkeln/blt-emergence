import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .mod import RoPE, EncoderBlock


# TODO: finish patching transformer
# INFO: each cell is a discrete state
# num_embeddings (otherwise, number of unique states in the lattice):
# GoL -> 2 states so 2 unqiue embedding values
# MCST -> 4 states so 4 unique embedding values
# INFO: d_model selection
# keep size of embedding vector small but effective for attention mechanisms
# WARN: convert lattice to torch.long before learning
class PatchingTransformer(nn.Module):
    def __init__(
            self,
            num_embeddings: int = 2,
            d_model: int = 8,
            num_layers: int = 6,
            num_heads: int = 4
    ):
        super().__init__()
        self.input_embed = nn.Embedding(num_embeddings, embedding_dim=d_model)
        self.d_model = d_model
        self.num_embeddings = num_embeddings
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.rope = RoPE(d_model)
        self.transformer = nn.ModuleList([
            EncoderBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, num_embeddings)

    def forward(self, x):
        x = self.input_embed(x.long())
        x = rearrange(x, 'b h w d -> b d h w')
        x = self.rope(x)
        x = rearrange(x, 'b d h w -> b (h w) d')

        for block in self.transformer:
            x = block(x)

        logits = self.lm_head(x)
        return logits
