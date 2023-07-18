# base.py

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


# --------------------------------------------------------------------------------
#   FFA_Layer_Base
# --------------------------------------------------------------------------------


class FFA_Layer_Base (nn.Module):

    def __init__ (self):
        
        super().__init__()
        self.thr = nn.Parameter(th.zeros(1))
        
    def _reset_parameters (self):
        
        raise NotImplementedError
        
    def reset_parameters (self):
        
        with th.no_grad():
            self._reset_parameters()
            self.thr.zero_()
        
    def forward (self, x):
        
        raise NotImplementedError
    
    def goodness (self, o):
        
        return o.pow(2).mean(dim=1, keepdim=True)

    def goodness_thr (self, o):
    
        return self.goodness(o) - self.thr

    
# --------------------------------------------------------------------------------
#   FFA_Module_Base
# --------------------------------------------------------------------------------
    
    
class FFA_Module_Base (nn.Module):
    '''
    hyp example = {
        'Nclasses': 2,
        others... for the FFA_Layers
    }
    ''' 
 
    def __init__ (self, hyp):
        
        super().__init__()
        self.hyp = hyp
        self.layers = nn.ModuleList(self.get_layers())
        self.register_buffer('gather', th.ones(len(self.layers)))
        self.reset_parameters()
        
    def get_layers (self):
    
        # It must return a list of FFA_Layer_Base
        raise NotImplementedError
        
    def reset_parameters (self):
        
        with th.no_grad():
            for lay in self.layers:
                lay.reset_parameters()   
          
    def forward_to_lay (self, x, layer):

        for lay in self.layers[:layer]:
            x = lay(x)
            x = F.normalize(x) * sqrt(x.shape[1])
        return self.layers[layer](x)
        
    def forward_to_gather (self, x):
               
        g = []
        for lay in self.layers:
                x = lay(x)
                g.append(lay.goodness(x))
                x = F.normalize(x) * sqrt(x.shape[1])  
        return th.cat(g, dim=1)
        
    def get_all_possible_matches (self, x):
    
        # If x.shape = (N, M) and Nclasses = k, this function must return a (N*k, M) tensor
        raise NotImplementedError
            
    def forward (self, x):
               
        x = self.get_all_possible_matches(x)        
        g = self.forward_to_gather(x)
        goodness = th.mv(g, self.gather).reshape(-1, self.hyp['Nclasses'])
        # goodness[i,k] = goodness of the k-th class, for the i-th input
        return goodness
        
    def sum_normalized_goodness (self, x):
    
        return F.normalize(self(x), p=1, dim=1, eps=0)
        
    def predict_proba (self, x):
    
        return F.softmax(self(x), dim=1)
        
    def predict (self, x):

        return th.argmax(self(x), dim=1, keepdim=True)
        
    def TS_loop (self, TS_dl):

        E, n = 0., 0
        with th.no_grad():
            for x, y in TS_dl:
                E += (self.predict(x) != y).sum()
                n += x.shape[0]
        return E.item()/n