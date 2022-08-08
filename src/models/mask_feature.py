import torch
from torch import Tensor


class MaskFeature(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        out = torch.zeros_like(x)
        out[mask] = x[mask]
        return out

    @staticmethod
    def ratio_mask(mask: Tensor, ratio: float):
        r"""Modifies :obj:`mask` by setting :obj:`ratio` of :obj:`True`
        entries to :obj:`False`. Does not operate in-place.

        Args:
            mask (torch.Tensor): The mask to re-mask.
            ratio (float): The ratio of entries to keep.
        """
        n = int(mask.sum())
        out = mask.clone()
        out[mask] = torch.rand(n, device=mask.device) < ratio
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
