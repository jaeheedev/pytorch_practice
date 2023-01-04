import glob
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
import torchvision.transforms as tf
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import tqdm
from torch.optim.adam import Adam
import time

transforms = tf.Compose([
    tf.Resize(64),
    tf.CenterCrop(64),
    tf.ToTensor(),
    tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageFolder(
    root = "./GAN",
    transform=transforms
)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=4, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.disc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.disc(x)
    

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm')!=-1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

device = "cuda" if torch.cuda.is_available() else "cpu"

G = Generator().to(device)
G.apply(weights_init)

D = Discriminator().to(device)
D.apply(weights_init)

G_optim = Adam(G.parameters(), lr = 0.00008, betas = (0.5, 0.999))
D_optim = Adam(D.parameters(), lr = 0.00008, betas=(0.5, 0.999))

for epochs in range(50):
    iterator = tqdm.tqdm(enumerate(loader, 0), total=len(loader))
    
    for i, data in iterator:
        D_optim.zero_grad()
        
        label = torch.ones_like(data[1], dtype=torch.float32).to(device)
        label_fake = torch.ones_like(data[1], dtype=torch.float32).to(device)
        
        real = D(data[0].to(device))
        
        Dloss_real = nn.BCELoss()(torch.squeeze(real), label)
        Dloss_real.backward()
        
        noise = torch.randn(label.shape[0], 100, 1, 1, device = device)
        fake = G(noise)
        
        output = D(fake.detach())
        
        Dloss_fake = nn.BCELoss()(torch.squeeze(output), label_fake)
        Dloss_fake.backward()
        
        Dloss = Dloss_real + Dloss_fake
        D_optim.step()
        
        G_optim.zero_grad()
        output = D(fake)
        Gloss = nn.BCELoss()(torch.squeeze(output), label)
        Gloss.backward()
        
        G_optim.step()
        
        iterator.set_description(f"epoch:{epochs} iteration: {i} D_loss: {Dloss} G_loss: {Gloss}")
        
torch.save(G.state_dict(), "Generator.pth")
torch.save(D.state_dict(), "Discriminator.pth")

print(time.gmtime(time.time()))

with torch.no_grad():
    G.load_state_dict(
        torch.load("Generator.pth", map_location=device)
    )
    feature_vector = torch.randn(1, 100, 1, 1).to(device)
    pred = G(feature_vector).squeeze()
    pred = pred.permute(1, 2, 0).cpu().numpy()
    
    plt.imshow(pred)
    plt.title("predicted image")
    plt.show()