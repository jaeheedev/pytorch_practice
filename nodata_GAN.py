# knowlege distillation
import tqdm
import torch
import torch.nn as nn

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms import RandomHorizontalFlip, RandomCrop
from torchvision.transforms import Normalize
from torch.utils.data.dataloader import DataLoader
from torchvision.models.resnet import resnet34, resnet18

from torch.optim.adam import Adam
from torch.optim.sgd import SGD

import torch.nn.functional as F


transforms = Compose([
    RandomCrop((32, 32), padding=4),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean = (0.4914, 0.4822, 0.4465),
              std=(0.247, 0.243, 0.261))
])

training_data = CIFAR10(root = "./", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

teacher = resnet34(pretrained=False, num_classes=10)
teacher.to(device)

lr = 1e-5
optim = Adam(teacher.parameters(), lr=lr)

for epoch in range(30):
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        optim.zero_grad()
        
        preds = teacher(data.to(device))
        
        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()
        
        iterator.set_description(f"epoch: {epoch+1} loss: {loss.item()}")

torch.save(teacher.state_dict(), "teacher.pth")

teacher.load_state_dict(torch.load("teacher.pth", map_location=device))

num_corr = 0

with torch.no_grad():
    for data, label in test_loader:
        
        output = teacher(data.to(device))
        preds = output.data.max(1)[1]
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr
    print(f"Accuracy:{num_corr/len(test_data)}")
    

class Generator(nn.Module):
    def __init__(self, dims=256, channels = 3):
        super(Generator, self).__init__()
        self.l1 = nn.Sequential(nn.Linear(dims, 128*8*8))
        
        self.conv_blocks0 = nn.Sequential(nn.BatchNorm2d(128),)
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(channels, affine=False)
        )
        
    def forward(self, z):
        out = self.l1(z.view(z.shape[0], -1))
        out = out.view(out.shape[0], -1, 8, 8)
        
        out = self.conv_blocks0(out)
        out = nn.functional.interpolate(out, scale_factor=2)
        out = self.conv_blocks1(out)
        out = nn.functional.interpolate(out, scale_factor=2)
        out = self.conv_blocks2(out)
        return out
    
teacher = resnet34(pretrained=False, num_classes=10)
teacher.load_state_dict(torch.load("./teacher.pth", map_location=device))
teacher.to(device)
teacher.eval()

student = resnet18(pretrained=False, num_classes=10)
student.to(device)

generator = Generator()
generator.to(device)

G_optim = Adam(generator.parameters(), lr=1e-3)
S_optim = SGD(student.parameters(), lr = 0.1, weight_decay=5e-4, momentum=0.9)

for epoch in range(500):
    for _ in range(5):
        noise = torch.randn(256, 256, 1, 1, device=device)
        S_optim.zero_grad()
        fake = generator(noise).detach()
        teacher_output = teacher(fake)
        student_output = student(fake)
        
        S_loss = nn.L1Loss()(student_output, teacher_output.detach())
        
        print(f"epoch{epoch}: S_loss {S_loss}")
        
        S_loss.backward()
        S_optim.step()
        
    noise = torch.randn(256, 256, 1, 1, device=device)
    G_optim.zero_grad()
    
    fake = generator(noise)
    teacher_output = teacher(fake)
    student_output = student(fake)
    
    G_loss = -1 * nn.L1Loss()(student_output, teacher_output)
    
    G_loss.backward()
    G_optim.step()
    
    print(f"epoch{epoch}: G_loss {G_loss}")
    
torch.save(student.state_dict(), "student.pth")
    
num_corr = 0

student.load_state_dict(torch.load("student.pth", map_location=device))

with torch.no_grad():
    for data, label in train_loader:
        
        output = student(data.to(device))
        preds = output.data.max(1)[1]
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr
    
    print(f"Accuracy:{num_corr/len(training_data)}")
    
num_corr = 0

with torch.no_grad():
    for data, label in test_loader:
        output = student(data.to(device))
        preds = output.data.max(1)[1]
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr
    
    print(f"Accuracy:{num_corr/len(test_data)}")
