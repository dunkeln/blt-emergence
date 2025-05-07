import torch


def verify_patch_counts(tokenized_batches, original_batch, pad_id=None):
    """
    tokenized_batches: list of lists of tensors, shape [B][T] where each tensor is (M_t, L)
    original_batch: tensor of shape (B, T, H, W)
    pad_id: if you padded with a special pad_id, set it here to ignore pads
    """
    B, T, _, _ = original_batch.shape
    for b in range(B):
        for t in range(T):
            patches = tokenized_batches[b][t]           # (M_t, L)
            # reconstruct flat cells, dropping pad tokens if given
            flat_recon = patches.reshape(-1)
            if pad_id is not None:
                flat_recon = flat_recon[flat_recon != pad_id]

            # original flat
            flat_orig = original_batch[b, t].reshape(-1)
            # count unique
            u_r, c_r = torch.unique(flat_recon, return_counts=True)
            u_o, c_o = torch.unique(flat_orig,  return_counts=True)
            dict_r = dict(zip(u_r.tolist(), c_r.tolist()))
            dict_o = dict(zip(u_o.tolist(), c_o.tolist()))
            if dict_r != dict_o:
                print(f"Mismatch at batch {b}, time {t}:")
                print("  patched counts:", dict_r)
                print("  original counts:", dict_o)
                return False
    print("All counts match!")
    return True
