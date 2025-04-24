import torch
import torch.nn as nn
import torch.nn.functional as F

# class RoPE2D(nn.Module):
#     """
#     Applies 2D Rotary Positional Embeddings.
#     Assumes input of shape (B, L, d_model) where L = H*W.
#     Splits the embedding into two halves: one for vertical (row) and one for horizontal (column) positions.
#     """
#     def __init__(self, d_model, height, width):
#         super(RoPE2D, self).__init__()
#         assert d_model % 2 == 0, "Model dimension must be even."
#         self.d_model = d_model
#         self.height = height
#         self.width = width
#         self.pos_y = torch.arange(height).unsqueeze(1)  # (H, 1)
#         self.pos_x = torch.arange(width).unsqueeze(1)   # (W, 1)
# 
#     def forward(self, x):
#         # x: (B, L, d_model), L = H * W.
#         B, L, d_model = x.shape
#         H, W = self.height, self.width
#         x = x.view(B, H, W, d_model)
#         d_half = d_model // 2
#         x_y, x_x = x[..., :d_half], x[..., d_half:]
#         
#         inv_freq_y = 1.0 / (10000 ** (torch.arange(0, d_half, 2, device=x.device, dtype=torch.float) / d_half))
#         inv_freq_x = 1.0 / (10000 ** (torch.arange(0, d_half, 2, device=x.device, dtype=torch.float) / d_half))
#         
#         pos_y = self.pos_y.float()  # (H, 1)
#         pos_x = self.pos_x.float()  # (W, 1)
#         
#         sinusoid_y = torch.einsum("i,j->ij", pos_y.squeeze(-1), inv_freq_y)  # (H, d_half/2)
#         sinusoid_x = torch.einsum("i,j->ij", pos_x.squeeze(-1), inv_freq_x)  # (W, d_half/2)
#         
#         sin_y, cos_y = torch.sin(sinusoid_y), torch.cos(sinusoid_y)
#         sin_x, cos_x = torch.sin(sinusoid_x), torch.cos(sinusoid_x)
#         
#         def apply_rotary(tensor, sin, cos):
#             # tensor: (B, H, W, d_half)
#             B, H, W, d = tensor.shape
#             tensor = tensor.view(B, H, W, d // 2, 2)
#             sin = sin.unsqueeze(0).unsqueeze(2).unsqueeze(-1)  # (1, H, 1, d/2, 1)
#             cos = cos.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
#             x_even = tensor[..., 0]
#             x_odd  = tensor[..., 1]
#             out_even = x_even * cos - x_odd * sin
#             out_odd  = x_even * sin + x_odd * cos
#             out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
#             return out
#         
#         x_y = apply_rotary(x_y, sin_y, cos_y)
#         x_x = apply_rotary(x_x, sin_x, cos_x)
#         x = torch.cat([x_y, x_x], dim=-1).view(B, L, d_model)
#         return x


class RoPE2D(nn.Module):
    """
    Applies 2D Rotary Positional Embeddings.
    Assumes input of shape (B, L, d_model) where L = H*W.
    Splits the embedding into two halves: one for vertical (row) positions,
    and one for horizontal (column) positions.
    """
    def __init__(self, d_model, height, width):
        super(RoPE2D, self).__init__()
        assert d_model % 2 == 0, "Model dimension must be even."
        self.d_model = d_model
        self.height = height
        self.width = width
        # Register pos_y and pos_x as buffers so they are moved to the correct device
        self.register_buffer("pos_y", torch.arange(height).unsqueeze(1))  # shape: (H, 1)
        self.register_buffer("pos_x", torch.arange(width).unsqueeze(1))   # shape: (W, 1)

    def forward(self, x):
        # x: (B, L, d_model), L = H * W.
        B, L, d_model = x.shape
        H, W = self.height, self.width
        x = x.view(B, H, W, d_model)
        d_half = d_model // 2
        x_y, x_x = x[..., :d_half], x[..., d_half:]

        inv_freq_y = 1.0 / (10000 ** (torch.arange(0, d_half, 2, device=x.device, dtype=torch.float) / d_half))
        inv_freq_x = 1.0 / (10000 ** (torch.arange(0, d_half, 2, device=x.device, dtype=torch.float) / d_half))
        
        # Ensure pos_y and pos_x are on the same device as x
        pos_y = self.pos_y.float().to(x.device)  # shape: (H, 1)
        pos_x = self.pos_x.float().to(x.device)  # shape: (W, 1)
        
        sinusoid_y = torch.einsum("i,j->ij", pos_y.squeeze(-1), inv_freq_y)  # (H, d_half/2)
        sinusoid_x = torch.einsum("i,j->ij", pos_x.squeeze(-1), inv_freq_x)  # (W, d_half/2)
        
        sin_y, cos_y = torch.sin(sinusoid_y), torch.cos(sinusoid_y)
        sin_x, cos_x = torch.sin(sinusoid_x), torch.cos(sinusoid_x)
        
        def apply_rotary(tensor, sin, cos):
            # tensor: (B, H, W, d_half)
            B, H, W, d = tensor.shape
            tensor = tensor.view(B, H, W, d // 2, 2)
            sin = sin.unsqueeze(0).unsqueeze(2).unsqueeze(-1)  # (1, H, 1, d/2, 1)
            cos = cos.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
            x_even = tensor[..., 0]
            x_odd  = tensor[..., 1]
            out_even = x_even * cos - x_odd * sin
            out_odd  = x_even * sin + x_odd * cos
            out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
            return out
        
        x_y = apply_rotary(x_y, sin_y, cos_y)
        x_x = apply_rotary(x_x, sin_x, cos_x)
        x = torch.cat([x_y, x_x], dim=-1).view(B, L, d_model)
        return x

class CausalDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CausalDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, tgt_mask=None, tgt_key_padding_mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask,
                                        key_padding_mask=tgt_key_padding_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x

class AutoregressiveLM(nn.Module):
    """
    A simple autoregressive LM that:
      1. Maps input token indices (on a 2D lattice) to embeddings.
      2. Adds 2D RoPE positional information.
      3. Flattens the lattice and processes it with a causal transformer using CausalDecoderLayer.
      4. Projects hidden states to vocabulary logits.
      
    Input lattice shape: (B, 1, H, W)
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers, height, width):
        super(AutoregressiveLM, self).__init__()
        self.height = height
        self.width = width
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rope2d = RoPE2D(d_model, height, width)
        self.layers = nn.ModuleList([
            CausalDecoderLayer(d_model, nhead)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # x: (B, 1, H, W) where tokens are in the single channel.
        B, C, H, W = x.shape
        assert C == 1, "Input lattice must have 1 channel representing token indices."
        x = x.view(B, H * W)
        print("pre embedding")
        x = self.token_embedding(x)              # (B, L, d_model)
        print("token embedding success")
        print(x.size())
        x = self.rope2d(x)                         # (B, L, d_model)
        print("rope embedding success")
        x = x.transpose(0, 1)                      # (L, B, d_model)
        L = x.size(0)
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        for layer in self.layers:
            x = layer(x, tgt_mask=causal_mask)
        x = x.transpose(0, 1)                      # (B, L, d_model)
        logits = self.lm_head(x)                   # (B, L, vocab_size)
        return logits
