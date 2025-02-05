import torch
import einops


def pack(x):
	B, N, *_ = x.shape
	return x.reshape(B * N, *_)


def unpack(x, s):
	B, *_ = x.shape
	return x.reshape(*s, *_)


def vector_gather(vectors, indices):
    """
    copied from https://gist.github.com/EricCousineau-TRI/cc2dc27c7413ea8e5b4fd9675050b1c0
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[N, L, D]
        indices: Tensor[N, K] or Tensor[N]
    Returns:
        Tensor[N, K, D] or Tensor[N, D]
    """
    N, L, D = vectors.shape
    squeeze = False
    if indices.ndim == 1:
        squeeze = True
        indices = indices.unsqueeze(-1)
    N2, K = indices.shape
    assert N == N2
    indices = einops.repeat(indices, "N K -> N K D", D=D)
    out = torch.gather(vectors, dim=1, index=indices)
    if squeeze:
        out = out.squeeze(1)
    return out
