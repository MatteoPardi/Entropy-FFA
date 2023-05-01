# model_selection.py

import numpy as np
from tqdm import tqdm
from types import FunctionType
from copy import deepcopy


# -------------------------------------------------------------------------
#   SearchCV_Object (Base Class)
# -------------------------------------------------------------------------


class SearchCV_Object:
    
    def __init__ (self, model_class, grid, device, N_points, N_trials_per_point):
        
        self.model_class = model_class
        self.grid = grid
        self.device = device
        self.N_points = N_points
        self.N_trials_per_point = N_trials_per_point
        self.results = []
     
    def reset (self):
        
        self.results = []
        
    def results_init (self):
        
        self.reset()
        for hyp in self.points():
            res = {'E(TR)': {}, 'E(VL)': {}}
            for key in res:
                res[key] = {'trials': [], 'mean': None, 'std': None}
            res['hyp'] = hyp
            self.results.append(res)
            
    def points (self):
        
        raise NotImplementedError
        
    def run (self, TR_dl, VL_dl, TR_pndl=None):
        
        if not self.results: self.results_init()
            
        for res in tqdm(self.results, total=self.N_points, desc=self.__class__.__name__):
                    
            for n in range(self.N_trials_per_point):
                M = self.model_class(res['hyp']).to(self.device)
                if TR_pndl: M.fit(TR_pndl)
                else: M.fit(TR_dl)
                Etr, Evl = M.TS_loop(TR_dl), M.TS_loop(VL_dl)
                res['E(TR)']['trials'].append(Etr)
                res['E(VL)']['trials'].append(Evl)
                
            for key in res:
                if key != 'hyp':
                    res[key]['mean'] = np.mean(res[key]['trials'])
                    if len(res[key]['trials']) > 1:
                        res[key]['std'] = np.std(res[key]['trials'], ddof=1)
        
        self.results.sort(key = lambda x : (x['E(VL)']['mean'], x['E(TR)']['mean']))
        
    def _get_readable_string (self, result):
    
        s = "hyp: {\n"
        for key, val in result['hyp'].items():
            s += " " + key + ": " + repr(val) + ",\n"
        s += "}\n"
        Ntrials = len(result['E(TR)']['trials'])
        if Ntrials > 1:
            s += f"E(VL) = {result['E(VL)']['mean']:.5g} +- " + \
                f"{result['E(VL)']['std']:.2g} " + \
                f"(sample size = {Ntrials:d})\n"
            s += f"E(TR) = {result['E(TR)']['mean']:.5g} +- " + \
                f"{result['E(TR)']['std']:.2g} " + \
                f"(sample size = {Ntrials:d})\n"
        else:
            s += f"E(VL) = {result['E(VL)']['mean']:.5g}\n"
            s += f"E(TR) = {result['E(TR)']['mean']:.5g}"
        return s
    
    def __str__ (self):
        
        if not self.results:
            s = "Method: " + self.__class__.__name__ + "\n"
            s += "Model: " + self.model_class.__name__
        if self.results:
            s = "The winner is:\n"
            s += self._get_readable_string(self.results[0])
        return s
    
    def __repr__ (self):
        
        return self.__str__()
        
    def save (self, path):
    
        with open(path, 'w') as file:
            file.write(80*'-')
            for result in self.results:
                s = self._get_readable_string(result)
                file.write('\n\n' + s + '\n\n' + 80*'-') 


# -------------------------------------------------------------------------        
#   GridSearchCV
# -------------------------------------------------------------------------              


class GridSearchCV (SearchCV_Object):
    
    def __init__ (self, model_class, grid, device=None, N_trials_per_point=1):
        
        for key, val in grid.items():
            if not isinstance(val, list): 
                raise Exception("GridSearchCV needs a grid with list as values.")
        N_points = np.prod([len(val) for key, val in grid.items()])
        if not device: device = th.device('cpu')
        super().__init__(model_class, grid, device, N_points, N_trials_per_point)
        
    def _rec_points (self, grid):
    
        keys = list(grid.keys())
        key = keys[0]
        if len(keys)==1:
            for val in grid[key]: 
                yield {key: val}
        else:
            grid_cutted = deepcopy(grid)
            del grid_cutted[key]
            for val in grid[key]: 
                this_hyp = {key: val}
                for hyp_cutted in self._rec_points(grid_cutted):
                    yield this_hyp | hyp_cutted
                    
    def points (self):
        
        for hyp in self._rec_points(self.grid):
            yield hyp
            
            
# -------------------------------------------------------------------------
#   RandomSearchCV
# -------------------------------------------------------------------------


class RandomSearchCV (SearchCV_Object):
    
    def __init__ (self, model_class, grid, device=None, N_points=20, N_trials_per_point=1):
        
        if not device: device = th.device('cpu')
        super().__init__(model_class, grid, device, N_points, N_trials_per_point)
        
    def points (self):
        
        for i in range(self.N_points):
            hyp = {}
            for key, val in self.grid.items():
                hyp[key] = val()     
            yield hyp
 
 
# ----------------- Utils for RandomSearchCV -----------------


class Uniform:

    def __init__ (self, low, high):
    
        self.low = low
        self.high = high
    
    def __call__ (self):
    
        return np.random.uniform(self.low, self.high)


class LogUniform:

    def __init__ (self, low, high):
    
        self.loglow = np.log(low)
        self.loghigh = np.log(high)
    
    def __call__ (self):
    
        return np.exp(np.random.uniform(self.loglow, self.loghigh))
        

class Choice:

    def __init__ (self, *args):
    
        self.elements = np.asarray(args)
    
    def __call__ (self):
    
        return np.random.choice(self.elements)
        
        
class Fixed:

    def __init__ (self, val):
    
        self.val= val
    
    def __call__ (self):
    
        return self.val