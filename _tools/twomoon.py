# twomoon.py

import numpy as np
from numpy.random import rand, randn
from numpy import pi, sqrt, cos, sin, concatenate
import matplotlib.pyplot as plt


class TwoMoon:
    
    def __init__ (self, noise=0.):
        
        self.c0 = (-0.5, -0.2)
        self.c1 = (+0.5, 0.2)
        self.width = 0.4
        self.a2 = (1 - self.width/2)**2
        self.b2 = (1 + self.width/2)**2
        self.noise = noise
        
        self.x = None
        self.y = None
        
    def class0_sample (self, N):
        
        angle = pi*rand(N)
        r = sqrt(self.a2 + rand(N)*(self.b2 - self.a2))
        x = (self.c0[0] + r*cos(angle) + self.noise*randn(N)).reshape(-1, 1)
        y = (self.c0[1] + r*sin(angle) + self.noise*randn(N)).reshape(-1, 1)
        return concatenate((x, y), axis=1), np.zeros((N, 1), dtype=int)
    
    def class1_sample (self, N):
        
        angle = pi*rand(N)
        r = sqrt(self.a2 + rand(N)*(self.b2 - self.a2))
        x = (self.c1[0] + r*cos(angle) + self.noise*randn(N)).reshape(-1, 1)
        y = (self.c1[1] - r*sin(angle) + self.noise*randn(N)).reshape(-1, 1)
        return concatenate((x, y), axis=1), np.ones((N, 1), dtype=int)
    
    def sample (self, N, size1=0.5):
        
        N1 = int(size1*N)
        N0 = N - N1
        x0, y0 = self.class0_sample(N0)
        x1, y1 = self.class1_sample(N1)
        x, y = concatenate((x0, x1)), concatenate((y0, y1))
        self.x = x
        self.y = y
        return x, y
    
    def grid_sample (self):
        
        x = np.linspace(-2, 2, 200)
        y = np.linspace(-1.5, 1.5, 200)
        x, y = np.meshgrid(x, y)
        x, y = x.flatten().reshape(-1, 1), y.flatten().reshape(-1, 1)
        return concatenate((x, y), axis=1)     
    
    def plot (self, x=None, y=None, moon=True):
        
        plt.figure()
        if x is not None and y is not None:
            x0 = x[np.argwhere(y.flatten() == 0).flatten()]
            x1 = x[np.argwhere(y.flatten() == 1).flatten()]
            plt.plot(x0[:,0], x0[:,1], ls='', marker='s', color='pink', alpha=1)
            plt.plot(x1[:,0], x1[:,1], ls='', marker='s', color='paleturquoise', alpha=1)
        if moon:
            x0 = self.x[np.argwhere(self.y.flatten() == 0).flatten()]
            x1 = self.x[np.argwhere(self.y.flatten() == 1).flatten()]
            plt.plot(x0[:,0], x0[:,1], ls='', marker='.', color='r')
            plt.plot(x1[:,0], x1[:,1], ls='', marker='.', color='b')  
        plt.tight_layout()

    def plot_proba (self, x=None, y=None, moon=True):
        
        plt.figure()
        if x is not None and y is not None:
            plt.scatter(x[:,0], x[:,1], marker='s', c=-y, cmap='Greys', alpha=1)
        if moon:
            x0 = self.x[np.argwhere(self.y.flatten() == 0).flatten()]
            x1 = self.x[np.argwhere(self.y.flatten() == 1).flatten()]
            plt.plot(x0[:,0], x0[:,1], ls='', marker='.', color='r')
            plt.plot(x1[:,0], x1[:,1], ls='', marker='.', color='b') 
        plt.tight_layout()