# module.py

from .base import FFA_Layer_Base, FFA_Module_Base
from math import sqrt
import torch as th
import torch.nn as nn


# --------------------------------------------------------------------------------
#   Fully_Connected_Layer
# --------------------------------------------------------------------------------


class Fully_Connected_Layer (FFA_Layer_Base):
    '''
    hyp example = {
        'A': (10, 20),
        'f_hid': nn.ReLU()
    }
    '''
    
    def __init__ (self, hyp):
        
        super().__init__()
        self.hyp = hyp
        self.layers = nn.ModuleList([nn.Linear(hyp['A'][0], hyp['A'][1]), hyp['f_hid']])      
        
    def _reset_parameters (self):
    
        for module in self.layers:
            for name, param in module.named_parameters():
                if name=='weight':
                    h = sqrt(3/param.shape[1])
                    param.uniform_(-h, +h)
                if name=='bias':
                    param.zero_()
                    
    def forward (self, x):
        
        for lay in self.layers:
            x = lay(x)
        return x


# --------------------------------------------------------------------------------
#   Fully_Connected_Module
# --------------------------------------------------------------------------------


class Fully_Connected_Module (FFA_Module_Base):
    '''
    hyp example = {
        'Nclasses': 2,
        'A': (5, 20, 20, 20),
        'f_hid': nn.ReLU()
        }
    ''' 
    
    def __init__ (self, hyp):
        
        super().__init__(hyp)
        self.register_buffer('eye', th.eye(hyp['Nclasses']))
        
    def get_layers (self):
    
        A, f_hid = self.hyp['A'], self.hyp['f_hid']
        return [Fully_Connected_Layer({'A': (A[i], A[i+1]), 'f_hid': f_hid}) for i in range(len(A)-1)]
        
    def get_all_possible_matches (self, x):
    
        k = self.hyp['Nclasses']
        x = th.cat((x.repeat_interleave(k, dim=0),
                    self.eye.repeat(x.shape[0], 1)),
                    dim=1)
        return x        