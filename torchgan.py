import os
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import HTML

import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils

import torchgan
from torchgan.models import *
from torchgan.losses import *
from torchgan.trainer import Trainer

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
print("Random Seed: ", manualSeed)

dataset = dsets.CIFAR100(
    root="./CIFAR100_data",
    train=True,
    transform=transforms.Compose(
        [
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ]
    ),
    download=True,
)

dataloader = data.DataLoader(dataset, batch_size=64, shuffle=True)

dcgan_network = {
    "generator": {
        "name" : DCGANGenerator,
        "args" : {
            "encoding_dims" : 100,
            "out_channels" : 3,
            "step_channels" : 32,
            "nonlinearity" : nn.LeakyReLU(0.2),
            "last_nonlinearity" : nn.Tanh(),
        },
        "optimizer" : {"name" : Adam, "args" : {"lr" : 0.0001, "betas" : (0.5, 0.999)}},
    },
    "discriminator": {
        "name" : DCGANDiscriminator,
        "args" : {
            "in_channels":3,
            "step_channels":32,
            "nonlinearity":nn.LeakyReLU(0.2),
            "last_nonlinearity":nn.LeakyReLU(0.2),
        },
        "optimizer":{"name":Adam, "args":{"lr":0.0003, "betas":(0.5, 0.999)}},
    },
}

minmax_losses = [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()]
wgangp_losses = [
    WassersteinGeneratorLoss(),
    WassersteinDiscriminatorLoss(),
    WassersteinGradientPenalty(),
]
lsgan_losses = [LeastSquaresGeneratorLoss(), LeastSquaresDiscriminatorLoss()]

real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(
    np.transpose(
        vutils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(), (1, 2, 0)
    )
)
plt.show()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.backends.cudnn.deterministic = True
    epochs = 50
else:
    device = torch.device("cpu")
    epochs = 5

print("Device: {}".format(device))
print("Epochs: {}".format(epochs))

trainer = Trainer(
    dcgan_network, lsgan_losses, sample_size=64, epochs=epochs, device=device
)

trainer(dataloader)

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(
    np.transpose(
        vutils.make_grid(
            real_batch[0].to(device)[:64], padding=5, normalize=True
        ).cpu(),
        (1, 2, 0),
    )
)

# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(plt.imread("{}/epoch{}_generator.png".format(trainer.recon, trainer.epochs)))
plt.show()

fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [
    [plt.imshow(plt.imread("{}/epoch{}_generator.png".format(trainer.recon, i)))]
    for i in range(1, trainer.epochs + 1)
]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

# Play the animation
HTML(ani.to_jshtml())