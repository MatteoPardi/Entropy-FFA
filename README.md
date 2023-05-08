# Experiments for "Entropy Based Regularization Improves Performance in Forward-Forward Algorithm"
Accompanying code to reproduce experiments from the paper "Entropy Based Regularization Improves Performance in Forward-Forward Algorithm", submitted to the European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2023).

## Paper's Abstract

**Authors**: Matteo Pardi, Domenico Tortorella and Alessio Micheli.

The forward-forward algorithm (FFA) is a recently proposed alternative to end-to-end backpropagation in deep neural networks.
FFA builds networks greedily layer by layer, thus being of particular interest in applications where memory and computational constraints are important.
In order to boost layers' ability to transfer useful information to subsequent layers, in this paper we propose a novel regularization term for the layer-wise loss function that is based on Renyi's quadratic entropy.
Preliminary experiments show accuracy is generally significantly improved across all network architectures.
In particular, smaller architectures become more effective in addressing our classification tasks compared to the original FFA.

## Guide

The project is separated in three subfolders, one for each dataset: *double moon*, *noisy double moon* and *MNIST*. Each subfolder contain similar files:
- Jupyter *Notebooks (already ran) to run experiments*. Each experiment has its own notebook, and in it all details about that particular experiment can be easily read, such as model selection or assessment strategy, or the learning curves. 
- A folder containing the *model selection results*.
- A folder containing the *final models*, saved as Pytorch models.
In various files, the names *T off* and *T on* are used to respectively refer to FFA and FFA+Entropy models.

The folder *_tools* contains tools needed to perform experiments: the FFA model, and useful classes to perform model selection or to manage tensor datasets. Moreover, here the file *twomoon.py* can be found, which is used to generate the datasets for double moon experiments (noisy and not noisy).

Tecnical details about methods used to perform experiments can be found in *Methods.md*.

## Contact Me

For any other additional information, you can email me at m.pardi3@studenti.unipi.it

## Copyright

```
Copyright (C) 2023, Matteo Pardi
Copyright (C) 2023, University of Pisa

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
