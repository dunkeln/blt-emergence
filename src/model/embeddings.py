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

