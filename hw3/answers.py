r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.1, 0.1, 0.015
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.1, 0.02, 0.005, 0.001, 0.001
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd, lr = 0.1, 0.0001
    # ========================
    return dict(wstd=wstd, lr=lr)

part2_q1 = r"""
**Answer:**
1. We expected no-dropout to have worse performance (both on train and test sets) than the models with dropout since we expected no-dropout model to overfit. 
However, we notice that no-dropout module performs better than dropout models on both train and test sets.
We think that these results can be explained by non-compatible **(wstd, lr)** hyperparameters, which can be fixed by further tweaking.
2. We expected low dropout values to have better results since more neurons are training, and the results for **dropout = {0.4}**
show better performance on the training and the test set than **dropout = {0.8}**. 
"""

part2_q2 = r"""
**Answer:**
Yes, it is possible. The accuracy measures how many correct predictions the
model produces, while the CE loss function measures the percentages the model
produces for the true labels. In other words, for a ground-truth label $y$,
predicted label $\hat{y}$ and percentage $p$ the model assigned to $y$, the accuracy
measures $\delta\left(y-\hat{y}\right)$ which is 0 or 1, while the CE measures $p$
which is between 0 and 1. For example, with a binary CE (two labels) and two
test samples, if the model produces percentages of $\left(0.49,1.00\right)$ for
the true labels then the accuracy is 50% and the loss is small; but if the
percentages are $\left(0.51,0.51\right)$ the accuracy increases to 100% and the
loss also increases.
Generally, the CE is a continuous “distance” function of the percentages the
model assigns to the GT labels, and the accuracy is basically a
Hamming-distance of the model predictions. As such, a group of outliers that
are far from the expected GT result can have a great effect on the loss and a
minor effect on the accuracy. A few epochs with such outliers can see an
increase to both the test loss and the test accuracy in each epoch. Is isn't
likely, but it is possible, and of course the likelihood decreases when the
size of the test set increases.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Answer TODO:**
- didn't explain why L=2 is best
**Answer:**
We see that network wasn't trainable for
$\left(K,L\right)\in\left\{\left(32,8\right),\left(32,16\right),\left(64,16\right)\right\}$.
For trainable configurations the difference in highest accuracy of different Ls
is small (for the same $K$), with a tiny advantage to lower $L$ values - the best
was $L=2$ and second best $L=4$. Also, the lower the $L$ the faster the learning.
The reason that some networks were not trainable is the ratio $L/K$ was too big.
Each Conv layer with kernel size $k=3$ produces an output where each “pixel” is
some weighted average of 9 input pixels, and the effect of a single input pixel
slowly propogates to farther output pixel when there more layers per block
(bigger $L$). For example, with $K=32$ and looking at the pixels $x_{1,1}$ and
$x_{32,3$2}$ (two opposite corners of the image feature map), then $x_{1,1}$ affects
as far as the $\left(15,15\right)$-th output pixel of the 8th Conv layer and
$x_{32,32}$ affects as far as the $\left(18,18\right)$-th output pixel. In the
output of the 16th Conv layer, already every coordinate of the output is
affected by every coordinate of $x$. At the end, we aren't looking at local
features anymore, but global ones. Recall that Conv layers are effective
because they enable the model to learn local features, hence the globalizing
effect of too large $L$ negates the effectiveness.
The reason that $L=2$ gave the best results on the test set is the fact that pooling layers are more 
common in that architecture which converse more local spatial invariancy for the features extracet by
the convolution layer and the fact that there far less learnable parameters in this architecture to
manage effective learning on relative small dataset such as the one we are using, as opposed to 
the deeper architectures that use pooling after deeper convolution-relu blocks and have far more
adjustable weights which may cause underfitting.
One option to resolve it is too increase $K$, and we see that it does help -
$K=32,L=8$ wasn't trainable but $K=64,L=8$ was.
A second option is to introduce pooling layers inside convolution blocks,
instead of just between convolution blocks.
Another option is to introduce dropout after Conv layers. Dropout introduces
redundancy into the network, which means in a single layer the same local
features are learned at different weights. When looking at the deeper layers,
they do weighted on a smaller number of distinct features, so the globalizing
effect explained earlier is smaller.
"""

part3_q2 = r"""
**Answer:**
All the networks were trainable, and as $K$ is larger than in the previous
experiment that is no surprise.
For $L=2$, the best result is for $K=64$. For $L=4$ and $L=8$, the best is
$K=2$56 with $K=128$ being a close second. The reason is that higher $K$
enables learning a bigger amount of local features (more filters), and larger
$L$ enables learning how the local features combine into less-local features.
The small $L=2$ doesn't allow enough depth for complicated combination of
refined local features, and instead favors combination of cruder local
features, which is why the best $L$ is 64 and not 128 or 256. The larger $L=4$
and $L=8$ allow more complicated combinations of more refined local features,
hence the larger $K$ values are better.
"""

part3_q3 = r"""
**Answer:**
No network was trainable, since the network is too deep to be trained without max-pooling
"""


part3_q4 = r"""
**Answer:**
The following changes were introduced to the network:
1. An nn.BatchNorm2d layer was added after each nn.Conv2d layer
2. An nn.Dropout layer with p=0.2 was added after each nn.Relu layer in the feature extractor block
3. An nn.Dropout layer with p=0.2 was added after each nn.Relu layer in the classifier block
Those changes were supposed to support better training since Dropout is known
to helping prevent overfit (introduce redundancy, effectively training more
models simultaneously) and BatchNorm is known to be a good technique for better
gradient flow (mitigates exploding/vanishing gradients, faster training).
All the networks were trained successfully, and achieved better results than
the original architecture. It isn't clear if the best results are produced by
$L=2$, $L=3$ or $L=4$, but $L=4$ had the least amount of overfit (or perhaps
was underfitted), suggesting more epochs could increase its test accuracy more
than the rest.
"""
# ==============

# ==============
# Part 4 answers


def part4_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq, temperature = 'ACT I. Yariv', 0.7
    # ========================
    return start_seq, temperature


part4_q1 = r"""
**Your answer:**

We split the corpus to sequences because:
- the entire corpus can't fit in GPU memory all at once

"""

part4_q2 = r"""
**Your answer:**



The text can show memory longer than the sequence length because:
- the hidden state isn't reset after `sequence_length` characters, and can remember further back.
- when training, we didn't reset the hidden state between batches either, just between epochs.

"""

part4_q3 = r"""
**Your answer:**

We do not shuffle the order of batches when training because:
- we assume a relation between the next character to the characters before it, and model it as a hidden state. If we would've shuffled the order, the "characters before the next character" would be random and the hidden state won't reflect text of a real work of art. As result, the network won't be able to learn correctly the parameters that control how the hidden state affects the output. (specifically $W_{hz}$, $W_{hr}$, $W_{hg}$, $W_{hy}$ and the biases)


"""

part4_q4 = r"""
**Your answer:**

1. During training we use a high temperature because we want the probability distribution of "what is the next character" to have a high variance. This allows the network to train against a wider range of predictions, promotes better learning and prevents overfitting.
We lower the temperature for sampling because it means a lower variance, and thus a better chance that the next generated character is actually related to the previous characters (represented as hidden state), as opposed to the next character being random and unrelated.
2. When the temperature is very high, the generated text contains many spelling mistakes and made-up words.
This is because the probability distribution is more uniform and has a higher variance.
Meaning, the next character generated has a higher chance to be unrelated to the previous characters.
Additionaly, the structure of the text looks more like a play because it has many line breaks and capital letters, and also more panctuation.
The has more of those because they are rarer than other characters (e.g. lowercase letters), and thus have a higher chance to be generated when the variance is high.
3. When the temperature is very low, the generated text contains almost zero spelling mistakes or made-up words, but the structure doesn't look like a play. The text also has a tendency to repeat an expression of 2-3 words several times in succession (longer sequences for lower temperatures) before breaking the loop and moving on to other words.
This is because the probability distribution is less uniform and has low variance, and thus is much more deterministic than before.
Basically, this is the opposite of the high-temperature case with parallel reasoning.
We do note that the lower variance supposedly could have caused more spelling mistakes, but this doesn't happen thanks to the memory contained in the hidden state being long enough (more than 3-4 characters back).
The repeating expressions can happen when the hidden state causes a "cycle" and is due to the deterministic nature of the distribution. For example, if the last characters were "the well " and we assume the network and hidden state are such that the most likely next character is "t", and afterwards "h", "e", " ", "w", etc. in a cycle, because of the low variance the most likely next character has a very high likelihood (delta-like) and the cycle will indeed be realized.
This results in the generated text containing a string of "the well the well the well" repeatedly, until the cycle breaks due to a lower-likelihood next character being generated (by chance).


"""
# ==============


# ==============
# Part 5 answers

PART5_CUSTOM_DATA_URL = None


def part5_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 6 answers

PART6_CUSTOM_DATA_URL = None


def part6_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=32, z_dim=128,
        data_label=1, label_noise=0.3,
        discriminator_optimizer=dict(
            type='Adam',
            weight_decay=0.02,
            betas=(0.5,0.999),
            lr=0.0002,
        ),
        generator_optimizer=dict(
            type='Adam',
            weight_decay=0.02,
            betas=(0.5,0.999),
            lr=0.0002,
        ),
    )
    # ========================
    return hypers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""