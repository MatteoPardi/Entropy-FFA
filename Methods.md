# Methods

In this file we describe tecnical details about how experiments are performed

## Minibatches of positive and negative data

Let's say we've a dataset $D = \\{ (\mathbf{x}_i, y_i) \\} _{i = 1}^N$ where $y_i \in \\{ 1, \ldots, C \\}$. To train models with SGD, minibatches of size 128 are sampled by boostrap (sampling with replacement) from $D$. Once a minibatch has been sampled, before to give it to the model, half of it (64 randomly chosen data) is made *negative*: these data are concatenated with 1-hot encodings of random wrong classes. The ramaining half of the minibatch is used as *positive*: these data are concatenated with 1-hot encodings of their own classes.

Conceptually, the entire dataset made by all possible positive and negative data has length $N \cdot C$. Since a *training epoch* is usually defined as a complete pass over the entire training set, in our experiments an epoch is defined as "$N \cdot C$ data have been presented to the model".

## Model Training

Stochastic gradient descent (SGD) with momentum is used as optimizer. As stopping criterium, we fix the number of epochs. For each experiment, hyperparameters defining SGD (learning rate $\eta$, momentum $\alpha$, number of epochs $N_{ep}$) have been selected performing a manual screening procedure. In double moon and noisy double moon experiment, a learning rate annealing schedule is used, since during the manual screening we noticed that in this way faster and better trainings are obtained. The annealing schedule used is an exponentially decaying schedule, from a bigger learning rate $\eta_\mathrm{hot}$ to a smaller one $\eta_\mathrm{cold}$ in $N_{ep}$ epochs.

## Model Selection

For FFA models, model selection is performed on weight decay $\lambda$, while for FFA+Entropy models on ($\lambda, T, s$), where $T$ is the *temperature* and $s$ the *kernel scale*. For details about explored ranges of hyperparameters, please refer to the jupyter notebooks dedicated to the experiments. 

A simple hold-out training/validation is used.

## Model Assessment

Once the model selection is finished, and the model has been selected, the final model is re-trained over the entire design set (training+validation) and the accuracy over the externel test set is computed. This is done 5 times, to get the final model test accuracy's mean and std over 5 trials.