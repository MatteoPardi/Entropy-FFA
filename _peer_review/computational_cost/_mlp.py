from math import sqrt
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


# --------------------------------------------------------------------------------
#   MLP_Module
# --------------------------------------------------------------------------------


class MLP_Module (nn.Module):
    '''
    hyp example = {
        'task': 'regression',
        'archit': (5, 20, 20, 1),
        'f_hid': nn.ReLU()
        }
    ''' 
    
    def __init__ (self, hyp):
        
        super().__init__()      
        self.hyp = self._check_task(hyp)
        self.layers = nn.ModuleList(self._get_layers())
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
    
        if self.hyp['task'] == 'binary classification': return th.sigmoid(self(x))
        if self.hyp['task'] == 'multiclass classification': return F.softmax(self(x), dim=-1)
        if self.hyp['task'] == 'regression': raise Exception("predict_proba in a regression task?")
                    
    def predict (self, x):
    
        if self.hyp['task'] == 'binary classification': return (self(x) > 0.).float()
        if self.hyp['task'] == 'multiclass classification': return self(x).argmax(dim=-1, keepdim=True)
        if self.hyp['task'] == 'regression': raise Exception("predict in a regression task?")
        
    def test (self, x, y):
    
        if self.hyp['task'].endswith('classification'): return (self.predict(x) != y).float().mean()
        if self.hyp['task'] == 'regression': return F.mse_loss(self(x), y)        
        
    def test_loop (self, TS_dl):

        E, n = 0., 0
        with th.no_grad():
            for x, y in TS_dl:
                E += self.test(x, y) * x.shape[0]
                n += x.shape[0]
        return E.item() / n
        
    def _check_task (self, hyp):
    
        if hyp['task'] not in ['regression', 'classification', 'multiclass classification', 'binary classification']:
            raise Exception(f"task {hyp['task']} is not recognized")
        if hyp['task'] == 'classification':
            if hyp['archit'][-1] == 1: hyp['task'] = 'binary classification'
            else: hyp['task'] = 'multiclass classification'
        return hyp
        
    def _get_layers (self):
        
        archit, f_hid = self.hyp['archit'], self.hyp['f_hid']
        layers = []
        for i in range(len(archit)-2):
            layers += [nn.Linear(archit[i], archit[i+1]), f_hid]
        layers += [nn.Linear(archit[-2], archit[-1])]
        return layers

    
# --------------------------------------------------------------------------------
#   MLP
# --------------------------------------------------------------------------------


class MLP (MLP_Module):
    '''
    hyp example = {
        'task': 'regression',
        'archit': (5, 20, 20, 1),
        'f_hid': nn.ReLU(),
        'weight_decay': 1e-3,
        'lr': 0.1,
        'momentum': 0.99,
        'n_epochs': 10
        }
    ''' 
    
    def __init__ (self, hyp):
        
        super().__init__(hyp)
        self.reset_optimizer()
        self.curve = []     
        self.requires_grad_(False)
        self._loss_fn = self._get_loss_fn()
        
    def reset_optimizer (self):
    
        # params without weight_decay (bias)
        bias = [param for name, param in self.named_parameters() if name[-4:]=='bias']  
        # params with weight_decay (weights)
        weights = [param for name, param in self.named_parameters() if name[-4:]!='bias']                     
        self.optim = th.optim.SGD([
            {'params': bias, 'weight_decay': 0.},
            {'params': weights}
            ],
            lr = self.hyp['lr'],
            momentum = self.hyp['momentum'],
            weight_decay = self.hyp['weight_decay']
        )   
    
    def fit (self, TR_dl, reset=True, verbose=False):
        
        if reset: 
            self.reset_parameters()
            self.reset_optimizer()
            if verbose: print(f"epoch 0: ", end='')
            self.curve = [self._loss_loop(TR_dl)]
            if verbose: print(f"L = {self.curve[-1]}")
        epoch = len(self.curve)
         
        self.requires_grad_(True)
        for epoch in range(epoch, epoch+self.hyp['n_epochs']):
            if verbose and not epoch % verbose: print(f"epoch {epoch}: ", end='')
            self.curve.append(self.train_loop(TR_dl))
            if verbose and not epoch % verbose: print(f"L = {self.curve[-1]}")
        self.requires_grad_(False)
        
    def train_loop (self, TR_dl):
        
        model_device = next(self.parameters()).device
        L_tot, n = 0., 0
        for x, y in TR_dl:
            x = x.to(model_device)
            y = y.to(model_device)   
            if self.hyp['task'] == 'multiclass classification': y = y.flatten()
            L = self._loss_fn(self(x), y)
            self.optim.zero_grad(True)
            L.backward()
            self.optim.step()
            L_tot += L * x.shape[0]
            n += x.shape[0]
        return L_tot.item() / n
    
    def plot_curve (self, start=0):
        
        L = self.curve
        plt.figure()
        plt.xlabel('epoch')
        plt.ylabel('L')
        plt.plot(list(range(start, len(L))), L[start:])
        
    def _get_loss_fn (self):
    
        if self.hyp['task'] == 'binary classification': return F.binary_cross_entropy_with_logits
        if self.hyp['task'] == 'multiclass classification': return F.cross_entropy
        if self.hyp['task'] == 'regression': return F.mse_loss
        
    def _loss_loop (self, TR_dl):
        
        model_device = next(self.parameters()).device
        L_tot, n = 0., 0
        with th.no_grad():
            for x, y in TR_dl:
                x = x.to(model_device)
                y = y.to(model_device)
                if self.hyp['task'] == 'multiclass classification': y = y.flatten()                
                L_tot += self._loss_fn(self(x), y) * x.shape[0]
                n += x.shape[0]
        return L_tot.item() / n