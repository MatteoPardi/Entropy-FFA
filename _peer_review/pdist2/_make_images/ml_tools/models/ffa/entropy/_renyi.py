import torch as th
import torch.nn.functional as F
from math import log, pi


# ---------------------------------------------------------------------
#   renyi
# ---------------------------------------------------------------------


def renyi (x, sigma):

    N, d = x.shape
    sigma_squared = sigma**2 / N**(2/(d+4))    
    return - th.log(2*th.sum(th.exp(-F.pdist(x)**2/2/sigma_squared)) + N) + \
           d/2*log(2*pi*sigma_squared) + 2*log(N)
 
 
# ---------------------------------------------------------------------
#   renyi_scalar_silverman
# ---------------------------------------------------------------------


def renyi_scalar_silverman (x):

    N, d = x.shape
    sigma_squared = th.var(x, dim=0).mean() * (4/N/(2*d+1))**(2/(d+4)) + 1e-20  
    return - th.log(2*th.sum(th.exp(-F.pdist(x)**2/2/sigma_squared)) + N) + \
           d/2*log(2*pi*sigma_squared) + 2*log(N)

           
# ---------------------------------------------------------------------
#   renyi_scalar_silverman_no_grad
# ---------------------------------------------------------------------


def renyi_scalar_silverman_no_grad (x):

    N, d = x.shape
    with th.no_grad():
        sigma_squared = th.var(x, dim=0).mean() * (4/N/(2*d+1))**(2/(d+4)) + 1e-20
    return - th.log(2*th.sum(th.exp(-F.pdist(x)**2/2/sigma_squared)) + N) + \
           d/2*log(2*pi*sigma_squared) + 2*log(N)