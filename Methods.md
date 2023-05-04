# Methods

In this file we describe tecnical details about how experiments are performed.

## Normalization conventions

The probability that a data is positive is defined as $p = \mathrm{Sigmoid} (g - \theta)$, where $g$ is the *goodness* and $\theta$ a learnable threshold. In our code, the goodness $g$ is defined as the mean-of-squares

$$
g = \frac{1}{d} || \mathbf{h} ||^2 = \frac{1}{d} \sum_{j=1}^d h_j^2 \ .
$$

We decide to use the mean-of-squares since we expect $\theta \sim O(1)$ in this way. Thus we can reasonably initialize $\theta = 0$. 

If $\mathbf{h}^{(l)} \in \mathbb{R}^{(d)}$ is the hidden representation of layer $l$, and $\bar{\mathbf{h}}^{(l+1)} \mathbb{R}^{(d)}$ is the input of layer $l+1$, the normalization used is

$$
\bar{\mathbf{h}}^{(l+1)} = \sqrt{d} \cdot \frac{\mathbf{h}^{(l)}}{|| \mathbf{h}^{(l)} ||} \ .
$$

The factor $\sqrt{d}$ is useful to get $|| \bar{\mathbf{h}}^{(l)} ||^2 = d$, thus averagly each component $h_j^2 \sim 1$, as assumed by standard weight's initialization methods.

## Minibatches of positive and negative data

Let's say we've a training set $D = \\{ (\mathbf{x}_i, y_i) \\} _{i = 1}^N$ where $y_i \in \\{ 1, \ldots, C \\}$ and $C$ is the number of classes. To train models with SGD, minibatches of size 128 are sampled by boostrap (sampling with replacement) from $D$. Once a minibatch has been sampled, before to give it to the model, half of it (64 randomly chosen data) is made *negative*: these data are concatenated with 1-hot encodings of random wrong classes. The ramaining half of the minibatch is used as *positive*: these data are concatenated with 1-hot encodings of their own classes.

Conceptually, the entire dataset made by all possible positive and negative data has length $N \cdot C$. Since a *training epoch* is usually defined as a complete pass over the entire training set, in our experiments we say that an epoch has passed when $N \cdot C$ data have been presented to the model.

## Model Training

Stochastic gradient descent (SGD) with momentum is used as optimizer. As stopping criterium, we fix the number of epochs. For each experiment, hyperparameters defining SGD (learning rate $\eta$, momentum $\alpha$, number of epochs $N_{EP}$) have been selected performing a manual screening. In double moon and noisy double moon experiment, a learning rate annealing schedule is used, since during the manual screening we noticed that in this way faster and better trainings were obtained. An exponentially decaying schedule is used: $\eta$ goes from a bigger value $\eta_\mathrm{hot}$ to a smaller one $\eta_\mathrm{cold}$ in $N_{EP}$ epochs.

## Model Selection

For FFA models, model selection is performed on *weight decay* $\lambda$, while for FFA+Entropy models on ($\lambda, T, s$), where $T$ is the *temperature* and $s$ the *kernel scale*. For details about ranges of hyperparameters explored, please refer to the jupyter notebooks dedicated to the experiments. 

A simple hold-out training/validation is used. For each set of hyperparameters explored, accuracy's mean and std over $N_\mathrm{TRIALS}$ trials is computed, and the final model is the one with greater mean accuracy. In double moon and noisy double moon experiments, we set $N_\mathrm{TRIALS} = 3$ (trainings are fast, we can afford it), while in MNIST $N_\mathrm{TRIALS} = 1$.

## Model Assessment

Once the model selection is finished, the final model is re-trained over the entire design set (training+validation) and the accuracy over the external test set is computed. 5 trials are ran, to get accuracy's mean and std.