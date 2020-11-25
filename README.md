### Update 24-11-2020: the official implementation of DDN, compatible with more recent versions of PyTorch is now implemented in [adversarial-library](https://github.com/jeromerony/adversarial-library)

## About

Code for the article "Decoupling Direction and Norm for Efficient Gradient-Based L2 Adversarial Attacks and Defenses" (https://arxiv.org/abs/1811.09600), to be presented at CVPR 2019 (Oral presentation)


Implementation is done in PyTorch 0.4.1 and runs with Python 3.6+. The code of the attack is also provided on TensorFlow. This repository also contains an implementation of the C&W L2 attack in PyTorch (ported from Carlini's [TF version](https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py))

For PyTorch 1.1+, check the pytorch1.1+ branch (`scheduler.step()` moved).


## Installation

This package can be installed via pip as follows:

```pip install git+https://github.com/jeromerony/fast_adversarial```

## Using DDN to attack a model

```python
from fast_adv.attacks import DDN
attacker = DDN(steps=100, device=device)

adv = attacker.attack(model, x, labels=y, targeted=False)
```
 
Where ```model``` is a pytorch ``nn.Module`` that takes inputs ```x``` and outputs the pre-softmax activations (logits), ```x``` is a batch of images (N x C x H x W) and ```labels``` are either the true labels (for ```targeted=False```) or the target labels (for ```targeted=True```). Note: ```x``` is expected to be on the [0, 1] range: you can use ```fast_adv.utils.NormalizedModel``` to wrap any normalization, such as mean subtraction.

See the "examples" folder for a [python](https://github.com/jeromerony/fast_adversarial/blob/master/examples/mnist_example.py) and a [jupyter notebook](http://nbviewer.jupyter.org/github/jeromerony/fast_adversarial/blob/master/examples/mnist_noteboook_example.ipynb) example

## Adversarial training with DDN

The following commands were used to adversarially train the models:

MNIST:
```
python -m fast_adv.defenses.mnist --lr=0.01 --lrs=30 --adv=0 --max-norm=2.4 --sn=mnist_adv_2.4
```

CIFAR-10 (adversarial training starts at epoch 200):
```
python -m fast_adv.defenses.cifar10 -e=230 --adv=200 --max-norm=1 --sn=cifar10_wrn28-10_adv_1
```

### Adversarially trained models 

* MNIST: https://www.dropbox.com/s/9onr3jfsuc3b4dh/mnist.pth
* CIFAR10: https://www.dropbox.com/s/ppydug8zefsrdqn/cifar10_wrn28-10.pth


