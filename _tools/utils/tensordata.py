# tensordata.py

import torch as th
import numpy as np


# -------------------------------------------------------------------------
#   Tensor Dataset 
# -------------------------------------------------------------------------


class TDataset:
    
    def __init__ (self, x, y):
        
        if x.shape[0] != y.shape[0]:
            raise Exception("x.shape[0] == y.shape[0] must be True")
        if x.device != y.device:
            raise Exception("x.device == y.device must be True")
        self.x = x
        self.y = y
        self.length = x.shape[0]
        self.indices = None # torch tensor 
        
    def __len__ (self):
        
        return self.length
    
    def __getitem__ (self, idx):
    
        if self.indices is not None: 
            return self.x[self.indices[idx]], self.y[self.indices[idx]]
        else:
            return self.x[idx], self.y[idx]
    
    def subset (self, idx):
    
        sub = TDataset(self.x, self.y)
        if self.indices is not None:  sub.indices = self.indices[idx]
        else: sub.indices = th.tensor(list(idx), device=self.x.device, dtype=th.long)
        sub.length = sub.indices.shape[0]
        return sub
    
    def dataloader (self, method=None, batch_size=None, drop_last=False):
    
        return TDataloader(self, method, batch_size, drop_last)

    def random_split (self, TR_size=0.8):
        
        TR_size = int(TR_size*len(self))
        perm = np.random.permutation(len(self)).tolist()
        TR, TS = self.subset(perm[:TR_size]), self.subset(perm[TR_size:])
        return TR, TS
 
    def __repr__ (self):
        
        return "Tensor Dataset"

    
# -------------------------------------------------------------------------
#   Tensor Dataloader
# -------------------------------------------------------------------------

    
class TDataloader:

    def __init__(self, tdataset, method=None, batch_size=None, drop_last=False):
        '''
        method: None or 'shuffle' or 'bootstrap'
        '''
        
        self.dataset = tdataset
        if method not in [None, 'shuffle', 'bootstrap']:
            raise Exception("method must be one of these: None, 'shuffle', 'bootstrap'")
        self.method = method
        if not batch_size: batch_size = len(self.dataset)
        self.batch_size = batch_size
        self.drop_last = drop_last
        n_batches, remainder = divmod(len(self.dataset), self.batch_size)
        if not self.drop_last and remainder > 0: n_batches += 1  
        self.n_batches = n_batches
        self.effective_length = n_batches*batch_size
        
    def __len__(self):
        
        return self.n_batches
        
    def __iter__(self):
        
        if self.method == 'bootstrap':
            idx = np.random.randint(0, len(self.dataset), size=self.effective_length).tolist()
        elif self.method == 'shuffle':
            idx = np.random.permutation(len(self.dataset)).tolist()
        if self.method:
            self.ready_dataset = self.dataset.subset(idx)
        else:
            self.ready_dataset = self.dataset
        self.i = 0
        return self

    def __next__(self):
        
        if self.i >= self.effective_length: raise StopIteration
        batch = self.ready_dataset[self.i:self.i+self.batch_size]
        self.i += self.batch_size
        return batch

    def __repr__ (self):
        
        return "Tensor Dataloader"


# -------------------------------------------------------------------------
#   Positive Negative Bootstrap Tensor Dataloader
# -------------------------------------------------------------------------
      
        
class PosNeg_Bootstrap_TDataloader:
    
    def __init__ (self, tdataset, batch_size=None):
        
        self.Nclasses = int(tdataset.y.max().item()) + 1 
        self.tot_length = len(tdataset)*self.Nclasses        
        self.dataset = tdataset
        if not batch_size: batch_size = self.tot_length
        self.batch_size = batch_size      
        n_batches, remainder = divmod(self.tot_length, self.batch_size)
        if remainder > 0: n_batches += 1  
        self.n_batches = n_batches
        self.effective_length = n_batches*batch_size
        self.device = tdataset.x.device
        self.eye = th.eye(self.Nclasses, device=self.device)
        self.mask = (th.ones(self.batch_size, device=self.device) > 0)
        self.mask[:int(self.batch_size/2)] = False
        
    def __len__ (self):
        
        return self.n_batches
        
    def __iter__ (self):
        
        idx = np.random.randint(0, len(self.dataset), size=self.effective_length).tolist()
        self.ready_dataset = self.dataset.subset(idx)
        self.i = 0
        return self 

    def __next__ (self):
    
        if self.i >= self.effective_length: raise StopIteration
        x, y = self.ready_dataset[self.i:self.i+self.batch_size]
        self.i += self.batch_size
        
        y = y.flatten()
        stay_positive = th.randperm(self.batch_size, device=self.device)
        stay_positive = self.mask[stay_positive]
        y_fake = th.randint(self.Nclasses-1, (y.shape[0],), device=self.device)
        y_fake = (y_fake >= y) + y_fake
        chosen_y = y_fake*(~stay_positive) + y*stay_positive
        ready_x = th.cat((x, self.eye[chosen_y]), dim=1)
        ready_y = stay_positive.float().reshape(-1, 1)
        return ready_x, ready_y

    def __repr__ (self):
        
        return "Postive Negative Bootstrap Tensor Dataloader"