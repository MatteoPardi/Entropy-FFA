import torch as th

# --------------------------------------------------------------
#    knn_avg
# --------------------------------------------------------------


def knn_avg (x, k=20, memory_limit=5e8):
    
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
                                                            th.sum(th.log(dist[:,:,1:k+1] + 1e-40), dim=(1,2))                                  
            del dist            
            n_batch += c2
        b_batch += c1       
    h = d/n/k*sum_log_kth_dist # + const indipendent from x
    if more_batches_are_present: return h
    else: return h[0]


# --------------------------- utils ---------------------------

def _get_k (k, n, d):

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