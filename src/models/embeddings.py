import torch
import torch.nn as nn

class EmbeddingWrapper(nn.Module):
    """
    wrapper for embeddings to accept iinput tensor
    a factory function
    """
    def __init__(self, embedding_class, **kwargs):
        super(EmbeddingWrapper, self).__init__()
        self.embedding = embedding_class(**kwargs)

    def forward(self, x):
        dim = x.size(1)
        return self.embedding(dim)

class RopeEmbedding(nn.Module):
    def __init__(self, d_model, max_dim, base=10_000):
        super(RopeEmbedding, self).__init__()
        assert d_model % 2 == 0, "d_model should be even"
        self.d_model = d_model
        self.d_half = d_model // 2
        self.max_dim = max_dim
        self.base = base

        inv_freq_row = 1.0 / (base ** (torch.arange(0, self.d_half, 2).float() / self.d_half))
        inv_freq_col = 1.0 / (base ** (torch.arange(0, self.d_half, 2).float() / self.d_half))
        self.inv_freq_row = inv_freq_row
        self.inv_freq_col = inv_freq_col

    def forward(self, dim: int):
        row_pos = torch.arange(dim).unsqueeze(1)
        col_pos = torch.arange(dim).unsqueeze(1)

        angles_row = row_pos * self.inv_freq_row.unsqueeze(0)
        angles_col = col_pos * self.inv_freq_col.unsqueeze(0)

        sin_row = angles_row.sin()
        cos_row = angles_row.cos()
        sin_col = angles_col.sin()
        cos_col = angles_col.cos()

        # Interleave cosine and sine for each axis.
        row_embed = torch.stack((cos_row, sin_row), dim=-1).reshape(dim, self.d_half)
        col_embed = torch.stack((cos_col, sin_col), dim=-1).reshape(dim, self.d_half)

        # Expand row and column embeddings to a (dim, dim, d_half) grid.
        row_embed_expanded = row_embed.unsqueeze(1).expand(dim, dim, self.d_half)
        col_embed_expanded = col_embed.unsqueeze(0).expand(dim, dim, self.d_half)

        # Concatenate along the last dimension to get a (dim, dim, d_model) tensor.
        pos_embed = torch.cat([row_embed_expanded, col_embed_expanded], dim=-1)
        return pos_embed


class StateEmbedding(nn.Module):
    def __init__(self, dim: int, d_model):
        super(StateEmbedding, self).__init__()
        self.linear = nn.Linear(dim, d_model)

    def forward(self, x):
        return self.linear(x)

class ComposeEmbeddings(nn.Module):
    def __init__(self, embeddings, combine_fn='sum'):
        """
        Args:
            embeddings (list of nn.Module): A list of embedding modules.
            combine_fn (str or callable): 'sum' to add outputs, 'concat' to concatenate, or a custom function.
        """
        super(ComposeEmbeddings, self).__init__()
        self.embeddings = nn.ModuleList(embeddings)
        self.combine_fn = combine_fn

    def forward(self, x):
        outputs = [emb(x) for emb in self.embeddings]
        match self.combine_fn:
            case "sum":
                combined = outputs[0]
                for out in outputs[1:]:
                    combined = combined + out
            case "concat":
                combined = torch.cat(outputs, dim=-1)
            case _:
                combined = torch.cat(outputs, dim=-1)
        return combined
