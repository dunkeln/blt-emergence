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

        # INFO: x & y freq
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, d_half, 2).float() / d_half))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor):
        _, D, H, W = x.size()
        assert D == self.d_model
        d_quarter = D // 4

        # INFO: split into x & y components
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
    def __init__(self, d_model: int = 8, num_heads: int = 4):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # linear projections for q, k, v
        self.in_proj = nn.Linear(d_model, 3 * d_model)
        # output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.fcn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # INFO: x: (B, seq_len, d_model)
        B, L, _ = x.shape

        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # INFO: scaled dot-product with built-in causal mask, aka flash attn.
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=True
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        attn_out = self.out_proj(attn_out)

        x = self.norm1(x + attn_out)
        fcn_out = self.fcn(x)
        return self.norm2(x + fcn_out)
