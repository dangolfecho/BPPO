import torch
import torch.nn as nn
from torch.distributions import Distribution

CONST_EPS = 1e-10


def orthogonal_initWeights(
    net: nn.Module,
    ) -> None:
    for e in net.parameters():
        if len(e.size()) >= 2:
            nn.init.orthogonal_(e)


def log_prob_func(
    dist: Distribution, action: torch.Tensor
    ) -> torch.Tensor:
    log_prob = dist.log_prob(action)
    if len(log_prob.shape) == 1:
        return log_prob
    else:
        return log_prob.sum(-1, keepdim=True)

def scale_action(a, low, high):
    """

    env action range -> [-1, 1]

    """
    return 2.0 * (a - low) / (high - low) - 1.0

def unscale_action(a, low, high):
    """

    [-1, 1] -> env action range

    """
    return low + (a + 1.0) * 0.5 * (high - low)
