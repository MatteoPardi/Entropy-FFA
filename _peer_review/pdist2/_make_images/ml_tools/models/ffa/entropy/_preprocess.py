import torch as th


def prescaling (x):

    if x.dim() == 1:
        x = x - x.mean()
        var = x.var() + 1e-12
        x = x / th.sqrt(var)
        log_det_var = th.log(var)
        return x, log_det_var/2
    if x.dim() > 1:
        x = x - x.mean(dim=-2, keepdim=True)
        var = x.var(dim=-2) + 1e-12
        x = x / th.sqrt(var[...,None,:])
        log_det_var = th.log(var).sum(dim=-1)
        return x, log_det_var/2


def presphering (x):

    if x.dim() == 1:
        return prescaling(x)
    if x.dim() > 1:
        x = x - x.mean(dim=-2, keepdim=True)
        cov = (x.transpose(-1,-2) @ x) / (x.shape[-2] - 1)
        U, s, _ = th.linalg.svd(cov)
        s = s + 1e-12
        x = x @ U / th.sqrt(s[...,None,:])
        log_det_cov = th.log(s).sum(dim=-1)
        return x, log_det_cov/2