import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class RoPE(nn.Module):
    def __init__(self, d_model: int = 8, theta_base: int = 10000):
        super().__init__()

        assert d_model % 4 == 0
        self.d_model = d_model
        d_half = d_model // 2

        # x & y freq.
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, d_half, 2).float() / d_half))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor):
        _, D, H, W = x.size()
        assert D == self.d_model
        d_quarter = D // 4

        # split into x & y components
        x_pos = torch.arange(H, device=x.device).float()
        y_pos = torch.arange(W, device=x.device).float()

        angles_x = torch.einsum("i,j->ij", x_pos, self.inv_freq)
        angles_y = torch.einsum("i,j->ij", y_pos, self.inv_freq)

        sin_x = angles_x.sin()[None, :, None, :]
        cos_x = angles_x.cos()[None, :, None, :]
        sin_y = angles_y.sin()[None, None, :, :]
        cos_y = angles_y.cos()[None, None, :, :]

        # INFO: 4-way splitting patches
        x = rearrange(x, "b (f d) h w -> b h w f d", f=4, d=d_quarter)
        x1, x2, x3, x4 = x.unbind(3)

        x_rot_x1 = x1 * cos_x - x2 * sin_x
        x_rot_x2 = x1 * sin_x + x2 * cos_x
        x_rot_y1 = x3 * cos_y - x4 * sin_y
        x_rot_y2 = x3 * sin_y + x4 * cos_y

        out = torch.stack([x_rot_x1, x_rot_x2, x_rot_y1, x_rot_y2], dim=3)
        out = rearrange(out, "b h w f d -> b (f d) h w")
        return out

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int=8, num_heads: int=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.fcn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def _causal_mask(self, x):
        N = x.size(1)
        mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False, attn_mask=self._causal_mask(x))
        x = self.norm1(x + attn_out)
        fcn_out = self.fcn(x)
        return self.norm2(x + fcn_out)

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

    def compute_entropy(self, logits: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        logp = torch.log(probs + 1e-12)
        entropy = -torch.sum(probs * logp, dim=-1)
        return entropy
