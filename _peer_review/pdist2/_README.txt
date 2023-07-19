Model assessment results of 'ffapdist2' comes from my master thesis work. The experimental setting is exactly the same used in the paper. The model selection was performed following these grids:

---------------------------------------
--- Double Moon & Noisy Double Moon ---
---------------------------------------

for d in [10, 100, 1000]:
    grids[f'ffapdist2{d:d}'] = RandomSearch({
        'n_classes': 2,
        'archit': (4, d, d, d),
        'f_hid': nn.ReLU(),
        'lr_hot': 0.3,
        'lr_cold': 0.001,
        'momentum': 0.98, 
        'weight_decay': LogUniform(1e-7, 1e-3),
        'temperature': LogUniform(1e-7, 1e-1),
        'entropy_method': 'pdist2',
        'n_epochs': 300
    }, n_points=32)
	
---------------------------------------
---             MNIST               ---
---------------------------------------

--------- Macro Grid ---------

utils[20] = { 
    'lr_hot': 0.01,
    'lr_cold': 0.01,
    'momentum': 0.99,
    'n_epochs': 24
}
utils[200] = { 
    'lr_hot': 0.05,
    'lr_cold': 0.05,
    'momentum': 0.99,
    'n_epochs': 24
}
utils[2000] = { 
    'lr_hot': 0.1,
    'lr_cold': 0.1,
    'momentum': 0.995,
    'n_epochs': 30
}

for d in [20, 200, 2000]:
    grid = {
        'n_classes': 10,
        'archit': (784+10, d, d, d),
        'f_hid': nn.ReLU(),
        # 'lr_hot': see utils...
        # 'lr_cold': see utils...
        # 'momentum': see utils...
        'weight_decay': [1e-8, 1e-6, 1e-4],
        'temperature': [1e-8, 1e-6, 1e-4, 1e-2],
        'entropy_method': 'pdist2'
        # 'n_epochs': see utils...
    }
    grids[f'ffapdist2{d:d}'] = GridSearch({**grid, **utils[d]})
	
--------- Micro Grid ---------

weight_decay is fixed: the value is the one found during the macro grid.
temperature is further explored: if T is the value found during the macro grid, then the micro grid is:
[T/10, T/4, T*4, T*10]


=======================================================================================================
Note. Even if in my master thesis the model is exactly the same used in the paper, the code changes a bit. To load the pdist2's pytorch models , in '_make_images' the folder 'ml_tools' contains the class used to produce those models. That's necessary to use th.load().