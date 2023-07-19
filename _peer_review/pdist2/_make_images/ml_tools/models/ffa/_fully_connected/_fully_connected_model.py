from ._fully_connected_module import Fully_Connected_Module
from .. import entropy
import numpy as np
import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
bcewl = F.binary_cross_entropy_with_logits


# --------------------------------------------------------------------------------
#   Fully_Connected_withEntropy
# --------------------------------------------------------------------------------


class Fully_Connected_withEntropy (Fully_Connected_Module):
    '''
    hyp example = {
        'n_classes': 2,
        'archit': (5, 20, 20, 20),
        'f_hid': nn.ReLU(),
        'lr_hot': 0.3,
        'lr_cold': 0.001,
        'momentum': 0.99, 
        'weight_decay': 1e-3,
        'temperature': 1,
        'entropy_method': 'knn',
        'k_neighbor': 'auto',
        'kernel_scale': 1,
        'n_epochs': 10
        }
    ''' 
    
    def __init__ (self, hyp):
        
        super().__init__(hyp)
        self.optim = None
        self.optim_scheduler = None
        self.curves = [{'L': [], 'H': [], 'Lreg': []}]*len(self.layers)        
        self.requires_grad_(False)
        
    def reset_optimizer (self, layer):
    
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
        lr_policy = LR_Exponential_Policy(self.hyp['lr_hot'], self.hyp['lr_cold'], self.hyp['n_epochs'])
        self.optim_scheduler = LambdaLR(self.optim, lr_lambda=lr_policy)
        
    def fit (self, TR_pndl, reset=True, verbose=False, measure_entropy=False):
        
        for layer in range(len(self.layers)):
            self.fit_layer(TR_pndl, layer, reset, verbose, measure_entropy)
        if verbose: print("done!")
          
    def fit_layer (self, TR_pndl, layer, reset=True, verbose=False, measure_entropy=False):
        
        if reset: 
            self.layers[layer].reset_parameters()  
            self.reset_optimizer(layer)
            self.curves[layer] = {'L': [], 'H': [], 'Lreg': []}
            if verbose: print(f"layer {layer}\n  epoch 0: ", end='')
            loss = self._loss_loop(TR_pndl, layer, measure_entropy)
            if verbose: print(f"(L, H, Lreg) = {loss}") if isinstance(loss, tuple) else print(f"L = {loss}")
            self._append_to_curves(layer, loss)
        
        epoch = len(self.curves[layer]['L'])
        self.layers[layer].requires_grad_(True)
        for epoch in range(epoch, epoch+self.hyp['n_epochs']):
            if verbose and not epoch % verbose: print(f"  epoch {epoch}: ", end='')
            loss = self.train_loop_layer(TR_pndl, layer, measure_entropy)
            if verbose and not epoch % verbose: print(f"(L, H, Lreg) = {loss}") if isinstance(loss, tuple) else print(f"L = {loss}")
            self.optim_scheduler.step()
            self._append_to_curves(layer, loss)
        self.layers[layer].requires_grad_(False)
    
    def train_loop_layer (self, TR_pndl, layer, measure_entropy=False):
        
        model_device = next(self.parameters()).device
        n = 0
        if measure_entropy: L_tot, H_tot, Lreg_tot = 0., 0., 0.
        else: L_tot = 0.
        
        for x, y in TR_pndl:
            
            x = x.to(model_device)
            y = y.to(model_device)            
            x = self.forward_to_layer(x, layer)
            L = bcewl(self.layers[layer].goodness_thr(x), y)
            if self.hyp['temperature']: 
                H = self._entropy_fn(x, layer)
                Lreg = L - self.hyp['temperature']*H
            else:
                Lreg = L
                if measure_entropy: 
                    with th.no_grad():
                        H = self._entropy_fn(x, layer)
            self.optim.zero_grad(True)
            Lreg.backward()
            self.optim.step()
            n += x.shape[0]
            L_tot += L * x.shape[0]
            if measure_entropy:
                H_tot += H * x.shape[0]
                Lreg_tot += Lreg * x.shape[0]
              
        L = L_tot.item() / n
        if measure_entropy:
            H = H_tot.item() / n
            Lreg = Lreg_tot.item() / n
            return L, H, Lreg
        else: return L
        
    def plot_curve (self, layer=None, start=0):
    
        if not layer:
            L = [curve['L'] for curve in self.curves]
            H = [curve['H'] for curve in self.curves]
            Lreg = [curve['Lreg'] for curve in self.curves]
            layers = range(len(L))
        else:
            layer -= 1
            L = [self.curves[layer]['L']]
            H = [self.curves[layer]['H']]
            Lreg = [self.curves[layer]['Lreg']]
            layers = [layer]   
        _H_and_Lreg_are_present = any(H) and any(Lreg)
 
        if _H_and_Lreg_are_present:
            fig, axs = plt.subplots(1, 3, figsize=(9.8, 3.5))
            y = [L, H, Lreg]
            name = ['L', 'H', 'L - T*H']
            for i in range(len(y)):
                axs[i].set_xlabel('epoch')
                axs[i].set_title(name[i])
                for j in range(len(layers)):
                    curve = y[i][j]
                    axs[i].plot(np.arange(len(curve))[start:], curve[start:], label=f'layer {layers[j]+1}')
                axs[i].legend(loc='best')
        else:
            plt.figure(figsize=(5, 3.5))
            plt.ylabel('L')
            plt.xlabel('epoch')
            for j in range(len(layers)):
                curve = L[j]
                plt.plot(np.arange(len(curve))[start:], curve[start:], label=f'layer {layers[j]+1}')
                plt.legend(loc='best')
        plt.tight_layout()
        
    def _entropy_fn (self, x, layer):
    
        if self.hyp['entropy_method'] == 'knn':
            return entropy.knn(x, k=self.hyp['k_neighbor'])
        if self.hyp['entropy_method'] == 'knn normalized':
            return entropy.knn(self.layers[layer].normalize(x), k=self.hyp['k_neighbor'])
        if self.hyp['entropy_method'] == 'renyi':
            return entropy.renyi(x, sigma=self.hyp['kernel_scale'])
        if self.hyp['entropy_method'] == 'renyi normalized':
            return entropy.renyi(self.layers[layer].normalize(x), sigma=self.hyp['kernel_scale'])
        if self.hyp['entropy_method'] == 'renyi scalar silverman':
            return entropy.renyi_scalar_silverman(x)
        if self.hyp['entropy_method'] == 'renyi scalar silverman normalized':
            return entropy.renyi_scalar_silverman(self.layers[layer].normalize(x))
        if self.hyp['entropy_method'] == 'knnavg':
            return entropy.knn_avg(x, k=self.hyp['k_neighbor'])
        if self.hyp['entropy_method'] == 'knnavg normalized':
            return entropy.knn_avg(self.layers[layer].normalize(x), k=self.hyp['k_neighbor'])
        if self.hyp['entropy_method'] == 'pdist2':
            return entropy.pdist2(x)
        if self.hyp['entropy_method'] == 'pdist2 normalized':
            return entropy.pdist2(self.layers[layer].normalize(x))
        if self.hyp['entropy_method'] == 'renyi scalar silverman no grad':
            return entropy.renyi_scalar_silverman_no_grad(x)
        if self.hyp['entropy_method'] == 'renyi scalar silverman no grad normalized':
            return entropy.renyi_scalar_silverman_no_grad(self.layers[layer].normalize(x))
        raise Exception("'entropy_method' not recognized")
 
    def _loss_loop (self, TR_pndl, layer, measure_entropy):
    
        model_device = next(self.parameters()).device
        n = 0
        if measure_entropy: L_tot, H_tot, Lreg_tot = 0., 0., 0.
        else: L_tot = 0.
        
        with th.no_grad():
            for x, y in TR_pndl:
            
                x = x.to(model_device)
                y = y.to(model_device)            
                x = self.forward_to_layer(x, layer)
                L = bcewl(self.layers[layer].goodness_thr(x), y)
                n += x.shape[0]
                L_tot += L * x.shape[0]
                if measure_entropy:
                    H = self._entropy_fn(x, layer)
                    Lreg = L - self.hyp['temperature']*H
                    H_tot += H * x.shape[0]
                    Lreg_tot += Lreg * x.shape[0]
                    
        L = L_tot.item() / n
        if measure_entropy:
            H = H_tot.item() / n
            Lreg = Lreg_tot.item() / n
            return L, H, Lreg
        else: return L
        
    def _append_to_curves (self, layer, loss):
    
        if isinstance(loss, tuple):
            self.curves[layer]['L'].append(loss[0])
            self.curves[layer]['H'].append(loss[1])
            self.curves[layer]['Lreg'].append(loss[2])
        else:
            self.curves[layer]['L'].append(loss)
 
# ------------------------------- utils -------------------------------
        
        
class LR_Exponential_Policy:
    '''
    Note: - cold_at_epoch = 5 ==> 5-th epoch is cold 
          - cold_at_epoch = 1 ==> always cold
    '''
    
    def __init__ (self, lr_hot, lr_cold, cold_at_epoch):
        
        self.ratio = lr_cold/lr_hot
        self.tau = cold_at_epoch - 1
        if self.tau: self.base = self.ratio**(1/self.tau)
        
    def __call__ (self, epoch):
        
        if epoch < self.tau: return self.base**epoch
        else: return self.ratio