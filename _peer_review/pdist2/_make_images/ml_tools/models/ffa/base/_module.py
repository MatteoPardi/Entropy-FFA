import torch as th
import torch.nn as nn
import torch.nn.functional as F

    
# --------------------------------------------------------------------------------
#   Module
# --------------------------------------------------------------------------------
    
    
class Module (nn.Module):
    '''
    hyp example = {
        'n_classes': 2,
        others... required by get_layers(hyp)
    }

    NotImplemented things:
        - get_layers(self, hyp)
        - get_all_possible_matches(self, x)
    ''' 
 
    def __init__ (self, hyp):
        
        super().__init__()
        self.hyp = hyp
        self.layers = nn.ModuleList(self._get_layers(hyp))
        self.register_buffer('gather', th.ones(len(self.layers)))
        if len(self.layers) > 1: self.gather[0] = 0. # exclude first layer's goodness 
        self.reset_parameters()
        
    def reset_parameters (self):
        
        with th.no_grad():
            for lay in self.layers: lay.reset_parameters()   
            
    def get_all_possible_matches (self, x):
    
        # If x.shape = (N, M) and Nclasses = k, this function must return a (N*k, M) tensor
        raise NotImplementedError
          
    def forward_to_layer (self, x, layer):

        for lay in self.layers[:layer]:
            x = lay.normalize(lay(x))
        return self.layers[layer](x)
        
    def layers_goodness (self, x):
               
        g = []
        for lay in self.layers:
                x = lay(x)
                g.append(lay.goodness(x))
                x = lay.normalize(x) 
        return th.cat(g, dim=1)
            
    def forward (self, x):
               
        x = self.get_all_possible_matches(x)        
        g = self.layers_goodness(x)
        goodness = th.mv(g, self.gather).reshape(-1, self.hyp['n_classes'])
        # goodness[i,k] = goodness of the k-th class, for the i-th input
        return goodness
        
    def sum_normalized_goodness (self, x):
    
        return F.normalize(self(x), p=1, dim=1, eps=0)
        
    def predict (self, x):

        return th.argmax(self(x), dim=1, keepdim=True)
        
    def predict_proba (self, x):
    
        return F.softmax(self(x), dim=1)
        
    def test (self, x, y):
    
        return (self.predict(x) != y).float().mean()
        
    def test_loop (self, TS_dl):

        E, n = 0., 0
        with th.no_grad():
            for x, y in TS_dl:
                E += self.test(x, y) * x.shape[0]
                n += x.shape[0]
        return E.item() / n
        
    def _get_layers (self, hyp):
    
        # It must return a list of ffa.base.Layer
        raise NotImplementedError