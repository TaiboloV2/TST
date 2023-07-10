import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------------------------------
class NormMinMax(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x -= x.min(dim=1, keepdim=True).values
        x /= x.max(dim=1, keepdim=True).values
        return x