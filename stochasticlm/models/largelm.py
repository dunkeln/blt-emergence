import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .mod import EncoderBlock, RoPE

class LatentTransformer(nn.Module):
    def __init__(self, d_model: int = 128, num_layers: int = 6, num_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, patch_embs: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        # patch_embs: (B, M, d_model)
        x = patch_embs
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask, is_causal=False)
        return self.norm(x)

class LocalDecoder(nn.Module):
    def __init__(self, d_model: int = 128, num_states: int = 2, max_patch_len: int = 256):
        super().__init__()
        self.max_len = max_patch_len
        self.head = nn.Linear(d_model, max_patch_len * num_states)
        self.num_states = num_states

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # latents: (B, M, d_model)
        B, M, _ = latents.shape
        out = self.head(latents)  # (B, M, L*V)
        return out.view(B, M, self.max_len, self.num_states)

class LargeLM(nn.Module):
    def __init__(
        self,
        num_states: int,
        d_model: int = 128,
        enc_heads: int = 4,
        latent_layers: int = 6,
        latent_heads: int = 8,
        max_patch_len: int = 256,
        device: str = "cpu"
    ):
        super().__init__()

        # INFO: pad_id = num_states
        self.d_model = d_model
        self.num_states = num_states
        self.max_patch_len = max_patch_len
        self.device = torch.device(device)
        self.input_embed = nn.Embedding(num_states + 1, d_model, padding_idx=num_states)
        self.rope = RoPE(d_model)
        self.encoder = EncoderBlock(d_model, enc_heads)
        self.latent = LatentTransformer(d_model, latent_layers, latent_heads)
        self.decoder = LocalDecoder(d_model, num_states, max_patch_len)
        self.num_embeddings = self.num_states
        self.num_layers = latent_layers
        self.num_heads = latent_heads

    def forward(self, cell_patches: torch.LongTensor, patch_mask: torch.BoolTensor) -> torch.Tensor:
        B, T, P, L = cell_patches.shape
        N = B * T
        patches = cell_patches.view(N, P, L).to(self.device)
        pmask   = patch_mask.view(N, P).to(self.device)
        flat = patches.reshape(N * P, L)

        emb = self.input_embed(flat)
        emb = rearrange(emb, 'np l d -> np d l 1')
        emb = self.rope(emb)
        emb = rearrange(emb, 'np d l 1 -> np l d')

        pad_mask = (flat == self.input_embed.padding_idx)

        enc = self.encoder(
            emb,
            is_causal=False,
            key_padding_mask=pad_mask
        )

        pooled = enc.mean(dim=1)
        patch_embs = pooled.view(N, P, self.d_model)

        kpm = ~pmask
        lat = self.latent(
            patch_embs,
            key_padding_mask=kpm
        )

        out = self.decoder(lat)
        _, P_out, L_out, V = out.shape
        return out.view(B, T, P_out, L_out, V)
