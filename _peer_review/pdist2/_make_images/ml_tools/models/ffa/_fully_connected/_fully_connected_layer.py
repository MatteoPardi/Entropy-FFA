from .. import base
from math import sqrt
import torch as th
import torch.nn as nn


# --------------------------------------------------------------------------------
#   Fully_Connected_Layer
# --------------------------------------------------------------------------------


class Fully_Connected_Layer (base.Layer):
    '''
    hyp example = {
        'archit': (10, 20),
        'f_hid': nn.ReLU()
    }
    '''
    
    def __init__ (self, hyp):
        
        super().__init__()
        self.hyp = hyp
        self.layers = nn.ModuleList([
            nn.Linear(hyp['archit'][0], hyp['archit'][1]),
            hyp['f_hid']
        ])
                    
    def forward (self, x):
        
        for lay in self.layers:
            x = lay(x)
        return x
        
    def _reset_parameters (self):
    
        for module in self.layers:
            for name, param in module.named_parameters():
                if name=='weight':
                    h = sqrt(3/param.shape[1])
                    param.uniform_(-h, +h)
                if name=='bias':
                    param.zero_()