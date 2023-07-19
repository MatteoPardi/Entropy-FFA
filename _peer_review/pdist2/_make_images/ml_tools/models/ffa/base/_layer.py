import torch as th
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


# --------------------------------------------------------------------------------
#   Layer
# --------------------------------------------------------------------------------


class Layer (nn.Module):
    '''
    NotImplemented things:
        - forward(self, x)
        - _reset_parameters(self)
    '''

    def __init__ (self):
        
        super().__init__()
        self.thr = nn.Parameter(th.zeros(1))
        
    def reset_parameters (self):
        
        with th.no_grad():
            self._reset_parameters()
            self.thr.zero_()
        
    def forward (self, x):
        
        raise NotImplementedError
        
    def normalize (self, o):
    
        return F.normalize(o) * sqrt(o.shape[1]) 
    
    def goodness (self, o):
        
        return o.pow(2).mean(dim=1, keepdim=True)

    def goodness_thr (self, o):
    
        return self.goodness(o) - self.thr
        
    def _reset_parameters (self):
        
        raise NotImplementedError