import numpy as np
import torch
from torch import nn


class GeneratorCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, label, gf, gb, df, db):
        np.mean(torch.linalg.norm((output - 1) ** 2))
        np.mean(torch.linalg.norm((output - 1) ** 2))
