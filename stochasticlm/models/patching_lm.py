import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .embeddings import RoPE


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
        # x: (B, seq_len, d_model)
        B, L, _ = x.shape

        # project to q, k, v and split heads
        qkv = self.in_proj(x)                             # (B, L, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)                    # each (B, L, d_model)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, L, hd)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # INFO: scaled dot-product with built-in causal mask, aka flash attn.
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=True
        )  # (B, nh, L, hd)

        # combine heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        attn_out = self.out_proj(attn_out)               # (B, L, d_model)

        # residual + norm + feed-forward
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

    def compute_entropy(self, logits: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        logp = torch.log(probs + 1e-12)
        entropy = -torch.sum(probs * logp, dim=-1)
        return entropy
