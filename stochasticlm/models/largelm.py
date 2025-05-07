import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import mlflow
from .mod import EncoderBlock, RoPE
from .patching_lm import PatchingTransformer

class LocalEncoder(nn.Module):
    def __init__(self, num_states, d_model=128, num_layers=2, num_heads=4):
        super().__init__()
        self.embed = nn.Embedding(num_states, d_model)
        self.rope  = RoPE(d_model)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, patch_seq):
        # patch_seq: (B, L)
        B, L = patch_seq.shape
        x = self.embed(patch_seq.long())          # (B, L, d)
        x = rearrange(x, 'b L d -> b d 1 L')
        x = self.rope(x)                          # (B, d, 1, L)
        x = rearrange(x, 'b d 1 L -> b L d')
        for blk in self.blocks:
            x = blk(x)
        emb = x.mean(dim=1)                       # (B, d)
        return self.norm(emb)

class LatentTransformer(nn.Module):
    def __init__(self, d_model=128, num_layers=6, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, patch_embs):
        # patch_embs: (B, M, d)
        x = patch_embs
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class LocalDecoder(nn.Module):
    def __init__(self, d_model=128, num_states=2, max_patch_len=256):
        super().__init__()
        self.max_len = max_patch_len
        self.head = nn.Linear(d_model, max_patch_len * num_states)
        self.num_states = num_states

    def forward(self, latents):
        # latents: (B, M, d)
        B, M, d = latents.shape
        out = self.head(latents)                      # (B, M, L*V)
        out = out.view(B, M, self.max_len, self.num_states)
        return out  # logits per-cell per-patch

class LargeLM(nn.Module):
    def __init__(self,
                 num_states: int,
                 d_model: int = 128,
                 enc_layers: int = 2,
                 enc_heads: int = 4,
                 latent_layers: int = 6,
                 latent_heads: int = 8,
                 max_patch_len: int = 256
    ):
        super().__init__()
        self.encoder = LocalEncoder(num_states, d_model, enc_layers, enc_heads)
        self.latent  = LatentTransformer(d_model, latent_layers, latent_heads)
        self.decoder = LocalDecoder(d_model, num_states, max_patch_len)

    def forward(self, cell_patches):
        # cell_patches: (B, M, L) padded to max_patch_len
        B, M, L = cell_patches.shape
        flat = cell_patches.view(B*M, L)
        emb  = self.encoder(flat)           # (B*M, d)
        emb  = emb.view(B, M, -1)           # (B, M, d)
        lat  = self.latent(emb)             # (B, M, d)
        logits = self.decoder(lat)          # (B, M, L, V)
        return logits
