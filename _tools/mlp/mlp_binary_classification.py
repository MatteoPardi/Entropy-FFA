# mlp_binary_classification.py

from math import sqrt
import numpy as np
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
BCEwL = F.binary_cross_entropy_with_logits


# --------------------------------------------------------------------------------
#   MLP_BinaryClassification_Module
# --------------------------------------------------------------------------------


def get_layers (A, f_hid):

    if A[-1] != 1:
        raise Exception("A[-1] == 1 must be True for binary classification")
    layers = []
    for i in range(len(A)-2):
        layers += [nn.Linear(A[i], A[i+1]), f_hid]
    layers += [nn.Linear(A[-2], A[-1])]
    return layers


class MLP_BinaryClassification_Module (nn.Module):
    '''
    hyp example = {
        'A': (5, 20, 20, 1),
        'f_hid': nn.ReLU()
        }
    ''' 
    
    def __init__ (self, hyp):
        
        super().__init__()
        self.hyp = hyp
        self.layers = nn.ModuleList(get_layers(hyp['A'], hyp['f_hid']))
        self.reset_parameters()
        
    def reset_parameters (self):
        
        with th.no_grad():
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
          
    def predict_proba (self, x):
    
        return th.sigmoid(self(x))
                    
    def predict (self, x):
        
        return (self(x) > 0.).long()

    
# --------------------------------------------------------------------------------
#   MLP_BinaryClassification
# --------------------------------------------------------------------------------


class LR_Policy:
    
    def __init__ (self, lr_hot, lr_cold, cold_at_epoch):
        
        self.ratio = lr_cold/lr_hot
        self.tau = cold_at_epoch - 1
        if self.tau: self.base = self.ratio**(1/self.tau)
        
    def __call__ (self, epoch):
        
        if epoch < self.tau: return self.base**epoch
        else: return self.ratio


class MLP_BinaryClassification (MLP_BinaryClassification_Module):
    '''
    hyp example = {
        'A': (5, 20, 20, 1),
        'f_hid': nn.ReLU(),
        'lr_hot': 0.1,
        'lr_cold': 0.001,
        'momentum': 0.99, 
        'weight_decay': 1e-3,
        'Nepoch': 10
        }
    ''' 
    
    def __init__ (self, hyp):
        
        super().__init__(hyp)
        self.optim = None
        self.optim_scheduler = None
        self.curve = []     
        self.requires_grad_(False)
    
    def TR_loop (self, TR_dl):
        
        model_device = next(self.parameters()).device
        n_batches = len(TR_dl)
        L_tot = 0.
        
        for x, y in TR_dl:
            
            x = x.to(model_device)
            y = y.to(model_device).float()            
            x = self(x)
            L = BCEwL(x, y)
            self.optim.zero_grad(True)
            L.backward()
            self.optim.step()
            L_tot += L
              
        L = L_tot.item() / n_batches
        return L
    
    def fit (self, TR_dl, reset=True):
        
        curve = []
        if reset: self.reset_parameters()

        # params without weight_decay
        group0 = [param for name, param in self.named_parameters() if name[-4:] == 'bias']  
        # params with weight_decay
        group1 = [param for name, param in self.named_parameters() if name[-4:] != 'bias']                     
        self.optim = th.optim.SGD([
            {'params': group0, 'weight_decay': 0.},
            {'params': group1}
            ],
            lr = self.hyp['lr_hot'],
            momentum = self.hyp['momentum'],
            weight_decay = self.hyp['weight_decay']
        )
        lr_policy = LR_Policy(self.hyp['lr_hot'], self.hyp['lr_cold'], self.hyp['Nepochs'])
        self.optim_scheduler = LambdaLR(self.optim, lr_lambda=lr_policy)  
        
        self.requires_grad_(True)
        for Nep in range(self.hyp['Nepochs']):
            L = self.TR_loop(TR_dl)
            self.optim_scheduler.step()
            curve.append(L)
        if reset: self.curve = curve
        else: self.curve += curve
        self.requires_grad_(False)
            
    def TS_loop (self, TS_dl):

        E, n = 0., 0
        with th.no_grad():
            for x, y in TS_dl:
                E += (self.predict(x) != y).sum()
                n += x.shape[0]
        return E.item()/n
    
    def plot_curve (self, start=0):
        
        L = self.curve
        plt.figure()
        plt.xlabel('epoch')
        plt.ylabel('L')
        plt.plot(np.arange(len(L))[start:], L[start:])