import os
import numpy as np
import matplotlib.pyplot as plt
import unittest
import torch
import torchvision
import torchvision.transforms as tvtf
#
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

seed = 42

plt.rcParams.update({'font.size': 12})
test = unittest.TestCase()

import hw3.optimizers as optimizers
help(optimizers.Optimizer)

# Test VanillaSGD
torch.manual_seed(42)
p = torch.randn(500, 10)
dp = torch.randn(*p.shape)*2
params = [(p, dp)]

vsgd = optimizers.VanillaSGD(params, learn_rate=0.5, reg=0.1)
vsgd.step()

expected_p = torch.load('tests/assets/expected_vsgd.pt')
diff = torch.norm(p-expected_p).item()
print(f'diff={diff}')
test.assertLess(diff, 1e-3)

import hw3.training as training
help(training.Trainer)

data_dir = os.path.expanduser('~/.pytorch-datasets')
ds_train = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=True, transform=tvtf.ToTensor())
ds_test = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=False, transform=tvtf.ToTensor())

print(f'Train: {len(ds_train)} samples')
print(f'Test: {len(ds_test)} samples')

import hw3.blocks as blocks
import hw3.models as models
import hw3.answers as answers
from torch.utils.data import DataLoader

# Overfit to a very small dataset of 20 samples
batch_size = 10
max_batches = 2
dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)

# Get hyperparameters
hp = answers.part2_overfit_hp()

torch.manual_seed(seed)

# Build a model and loss using our custom MLP and CE implementations
model = models.MLP(3*32*32, num_classes=10, hidden_features=[128]*3, wstd=hp['wstd'])
loss_fn = blocks.CrossEntropyLoss()

# Use our custom optimizer
optimizer = optimizers.VanillaSGD(model.params(), learn_rate=hp['lr'], reg=hp['reg'])

# Run training over small dataset multiple times
trainer = training.BlocksTrainer(model, loss_fn, optimizer)
best_acc = 0
for i in range(20):
    res = trainer.train_epoch(dl_train, max_batches=max_batches)
    best_acc = res.accuracy if res.accuracy > best_acc else best_acc
    
test.assertGreaterEqual(best_acc, 98)
