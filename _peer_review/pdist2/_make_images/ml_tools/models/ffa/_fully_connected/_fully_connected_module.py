from .. import base
from ._fully_connected_layer import Fully_Connected_Layer
import torch as th
import torch.nn as nn


# --------------------------------------------------------------------------------
#   Fully_Connected_Module
# --------------------------------------------------------------------------------


class Fully_Connected_Module (base.Module):
    '''
    hyp example = {
        'n_classes': 2,
        'archit': (5, 20, 20, 20),
        'f_hid': nn.ReLU()
        }
    ''' 
    
    def __init__ (self, hyp):
        
        super().__init__(hyp)
        self.register_buffer('eye', th.eye(hyp['n_classes']))
        
    def get_all_possible_matches (self, x):
    
        k = self.hyp['n_classes']
        x = th.cat((x.repeat_interleave(k, dim=0),
                    self.eye.repeat(x.shape[0], 1)),
                    dim=1)
        return x 

    def _get_layers (self, hyp):
    
        A, f_hid = hyp['archit'], hyp['f_hid']
        return [Fully_Connected_Layer({'archit': (A[i], A[i+1]), 'f_hid': f_hid}) for i in range(len(A)-1)]        