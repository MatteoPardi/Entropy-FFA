import torch as th
import torch.nn.functional as F


# ---------------------------------------------------------------------
#   pdist2
# ---------------------------------------------------------------------


def pdist2 (x):

    return th.mean(F.pdist(x)**2)