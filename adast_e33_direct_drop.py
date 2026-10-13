import torch


def direct_drop_forward_compact(pristine_patch, keep_mask):
    """
    Frozen-checkpoint direct token dropping.

    Preserves:
      - Original spatial/temporal selection mask
      - Selected-token order
      - Physical token count

    Removes:
      - Representation consolidation / merging
    """
    if pristine_patch.size(0) != 1:
        raise RuntimeError("E33 requires batch size 1")

    idx = torch.nonzero(
        keep_mask[0] > 0.5,
        as_tuple=False
    ).squeeze(1)

    compact = pristine_patch.index_select(1, idx)

    return compact, idx
