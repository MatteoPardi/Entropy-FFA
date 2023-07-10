# model.py

from .module import Fully_Connected_Module
from math import pi, log
import numpy as np
import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
BCEwL = F.binary_cross_entropy_with_logits


# --------------------------------------------------------------------------------
#   FFA_withEntropy
# --------------------------------------------------------------------------------


class LR_Policy:
    
    def __init__ (self, lr_hot, lr_cold, cold_at_epoch):
        
        self.ratio = lr_cold/lr_hot
        self.tau = cold_at_epoch - 1
        if self.tau: self.base = self.ratio**(1/self.tau)
        
    def __call__ (self, epoch):
        
        if epoch < self.tau: return self.base**epoch
        else: return self.ratio


class FFA_withEntropy (Fully_Connected_Module):
    '''
    hyp example = {
        'Nclasses': 2,
        'A': (5, 20, 20, 20),
        'f_hid': nn.ReLU(),
        'lr_hot': 0.3,
        'lr_cold': 0.001,
        'momentum': 0.99, 
        'weight_decay': 1e-3,
        'temperature': 1,
        'kernel_scale' = 1,
        'Nepochs': 10
        }
    ''' 
    
    def __init__ (self, hyp):
        
        super().__init__(hyp)
        if len(hyp['A']) > 2: self.gather[0] = 0. # don't include first layer goodness 
        self.optim = None
        self.optim_scheduler = None
        self.curves = [None]*len(self.layers)        
        self.requires_grad_(False)
    
    def renyi_quadratic_entropy (self, x):
    
        N, d = x.shape
        sigma_squared = self.hyp['kernel_scale']**2 / N**(2/(d+4))
        return - th.log(2*th.sum(th.exp(-F.pdist(x)**2/2/sigma_squared)) + N) + \
               d/2*log(2*pi*sigma_squared) + 2*log(N)
    
    def TR_loop_1lay (self, TR_pndl, layer):
        
        model_device = next(self.parameters()).device
        n_batches = len(TR_pndl)
        L_tot, H_tot, Lreg_tot = 0., 0., 0.
        
        for x, y in TR_pndl:
            
            x = x.to(model_device)
            y = y.to(model_device)            
            x = self.forward_to_lay(x, layer)
            L = BCEwL(self.layers[layer].goodness_thr(x), y)
            if self.hyp['temperature']: 
                H = self.renyi_quadratic_entropy(x)
                Lreg = L - self.hyp['temperature']*H
            else:
                Lreg = L
                with th.no_grad():
                    H = self.renyi_quadratic_entropy(x)
            self.optim.zero_grad(True)
            Lreg.backward()
            self.optim.step()
            L_tot += L
            H_tot += H
            Lreg_tot += Lreg
              
        L = L_tot.item() / n_batches
        H = H_tot.item() / n_batches
        Lreg = Lreg_tot.item() / n_batches 
        return L, H, Lreg
    
    def fit_1lay (self, TR_pndl, layer, reset=True, verbose=False):
        
        curve = {'L': [], 'H': [], 'Lreg': []}
        if reset: self.layers[layer].reset_parameters()
            
        # params without weight_decay
        group0 = [param for name, param in self.layers[layer].named_parameters() \
                  if (name == 'thr' or name[-4:] == 'bias')]  
        # params with weight_decay
        group1 = [param for name, param in self.layers[layer].named_parameters() \
                  if not (name == 'thr' or name[-4:] == 'bias')]                     
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
        
        self.layers[layer].requires_grad_(True)
        for Nep in range(self.hyp['Nepochs']):
            if verbose: print(f"  epoch {Nep+1} running...", end='')
            L, H, Lreg = self.TR_loop_1lay(TR_pndl, layer)
            self.optim_scheduler.step()
            curve['L'].append(L)
            curve['H'].append(H)
            curve['Lreg'].append(Lreg)
            if verbose: print(f"  L = {L}")
        if reset: self.curves[layer] = curve
        else: 
            for key, val in self.curves[layer].items():
                val += curve[key]
        self.layers[layer].requires_grad_(False)
    
    def fit (self, TR_pndl, reset=True, verbose=False):
        
        for layer in range(len(self.layers)):
            if verbose: print(f"layer {layer}")
            self.fit_1lay(TR_pndl, layer, reset, verbose)
        if verbose: print("done!")
        
    def plot_curve (self, layer=None, start=0):
    
        if not layer:
            L = [curve['L'] for curve in self.curves]
            H = [curve['H'] for curve in self.curves]
            Lreg = [curve['Lreg'] for curve in self.curves]
        else:
            L = self.curves[layer]['L']
            H = self.curves[layer]['H']
            Lreg = self.curves[layer]['Lreg']   
        fig, axs = plt.subplots(1, 3, figsize=(9.8, 2.5))
        n_layers = len(L)    
        y = [L, H, Lreg]
        name = ['L', 'H', 'L - T*H']
        for i in range(len(y)):
            axs[i].set_xlabel('epoch')
            axs[i].set_title(name[i])
            for layer in range(n_layers):
                curve = y[i][layer]
                axs[i].plot(np.arange(len(curve))[start:], curve[start:], label=f'layer {layer+1}')
            axs[i].legend(loc='best')
        plt.tight_layout()