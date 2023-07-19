import numpy as np
import torch as th
from scipy.special import gammaln, digamma
from ._preprocess import prescaling, presphering


# --------------------------------------------------------------
#    knn
# --------------------------------------------------------------


def knn (x, k='auto', preprocess=None, memory_limit=1e8):

    x, log_det_preprocess = _preprocess(x, preprocess)
    return _knn(x, k, memory_limit) + log_det_preprocess


def _knn (x, k='auto', memory_limit=1e8):
    
    more_batches_are_present = False
    if x.dim() == 3: more_batches_are_present = True
    elif x.dim() == 2: x = x[None,:,:]
    elif x.dim() == 1: x = x[None,:,None]
    else: raise Exception("x must be 1D x[n], or 2D x[n,d], or 3D (batches) x[b,n,d]")
    b, n, d = x.shape
    k = _get_k(k, n, d)
    # b x n x n tensors could be too big. Batches of c1 x c2 x n are safer.
    c1, c2 = _get_c1_c2(b, n, memory_limit)

    sum_log_kth_dist = th.zeros(b, device=x.device)
    b_batch = 0
    while b_batch < b:
        n_batch = 0
        while n_batch < n:
            x_batch = x[b_batch:b_batch+c1, n_batch:n_batch+c2] # x_batch[c1,c2,d]
            dist, _ = th.cdist(x_batch, x[b_batch:b_batch+c1]).sort(dim=2) # dist[c1,c2,n] (n-sorted)
            del _
            sum_log_kth_dist[b_batch:b_batch+c1] = sum_log_kth_dist[b_batch:b_batch+c1] + \
                                                            th.sum(th.log(dist[:,:,k] + 1e-40), dim=1)                                  
            del dist            
            n_batch += c2
        b_batch += c1       
    h = digamma(n) - digamma(k) + d/2*np.log(np.pi) - gammaln(d/2+1) + d/n*sum_log_kth_dist   
    if more_batches_are_present: return h
    else: return h[0]


# --------------------------- utils ---------------------------


def _preprocess (x, method=None):

    if method == 'prescaling': return _prescaling(x)
    elif method == 'presphering': return _presphering(x)
    elif method == None: return x, 0.
    else: raise Exception("preprocess must be in [None, 'prescaling', 'presphering']")
    

def _get_k (k, n, d):

    if isinstance(k, str): 
        if k == 'auto':
            # my semi-empirical rule
            k = round(n**min(0.3, 4/(4+d)))
        elif k in ['N**(4/(4+d))', 'N**(4/(d+4))']: 
            k = round(n**(4/(4+d)))
        else: 
            k = round(n**float(k[3:]))
    if k < 1: k = 1
    if k >= n: k = n - 1
    return k 


def _get_c1_c2 (b, n, memory_limit):
    
    size_of_float32 = 4 # bytes
    c2 = int(memory_limit / size_of_float32 / n)
    if c2 < 1: c2 = 1
    if c2 > n:
        c2 = n
        c1 = int(memory_limit / size_of_float32 / n / c2)
        if c1 < 1: c1 = 1
        if c1 > b: c1 = b
    else: c1 = 1
    return c1, c2    