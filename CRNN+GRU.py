import string
import matplotlib.pyplot as plt
import glob
import tqdm
from PIL import Image
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam

def get_BOW(corpus):
    # 공백 문자 <pad>를 0으로 설정
    BOW = {"<pad>":0}
    
    for letter in corpus:
        if letter not in BOW.keys():
            BOW[letter] = len(BOW.keys())
            
    return BOW

class Captcha(Dataset):
    def __init__(self, pth, train=True):
        self.corpus = string.ascii_lowercase + string.digits
        self.BOW = get_BOW(self.corpus)
        
        self.imgfiles = glob.glob(pth + "/*.png")
        
        self.train = train
        self.trainset = self.imgfiles[:int(len(self.imgfiles)*0.8)]
        self.testset = self.imgfiles[int(len(self.imgfiles)*0.8):]
    
    def get_seq(self, line):
        label = []
        print(line)
        
        for letter in line:
            label.append(self.BOW[letter])
        
        return label
    
    def __len__(self):
        if self.train:
            return len(self.trainset)
        else:
            return len(self.testset)
    
    def __getitem__(self, i):
        if self.train:
            data = Image.open(self.trainset[i]).convert("RGB")
            
            label = self.trainset[i].split("/")[-1]
            label = label.split(".png")[0]
            label = self.get_seq(label)
            
            data = np.array(data).astype(np.float32)
            
            data = np.transpose(data, (2, 0, 1))
            label = np.array(label)
            
            return data, label
        else:
            data = Image.open(self.testset[i]).convert("RGB")
            label = self.testset[i].split("/")[-1]
            label = label.split(".png")[0]
            label = self.get_seq(label)
            
            data = np.array(data).astype(np.float32)
            label = np.array(label)
            
            return data, label

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = (3,5), stride=(2,1)):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), padding=1)
        
        self.downsample = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x_ = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        x_ = self.downsample(x_)
        
        x += x_
        x = self.relu(x)
        
        return x

class CRNN(nn.Module):
    def __init__(self, output_size):
        super(CRNN, self).__init__()
        
        self.c1 = BasicBlock(in_channels=3, out_channels=64)
        self.c2 = BasicBlock(in_channels=64, out_channels=64)
        self.c3 = BasicBlock(in_channels=64, out_channels=64)
        self.c4 = BasicBlock(in_channels=64, out_channels=64)
        self.c5 = nn.Conv2d(64, 64, kernel_size=(2, 5))
        
        self.gru = nn.GRU(64, 64, batch_first = False)
        
        self.fc1 = nn.Linear(in_features=64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        
        x = x.view(x.shape[0], 64, -1)
        x = x.permute(2, 0, 1)
        
        x, _ = self.gru(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        x = F.log_softmax(x, dim=-1)
        
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = Captcha(pth = "data/CH12")
loader = DataLoader(dataset, batch_size = 8)

model = CRNN(output_size=len(dataset.BOW)).to(device)

optim = Adam(model.parameters(), lr = 0.0001)

for epoch in range(200):
    iterator = tqdm.tqdm(loader)
    
    for data, label in iterator:
        optim.zero_grad()
        preds = model(data.to(device))
        
        preds_size = torch.IntTensor([preds.size(0)] * 8).to(device)
        
        target_len = torch.IntTensor([len(txt) for txt in label]).to(device)
        
        loss = nn.CTCLoss(blank=0)(preds, label.to(device), preds_size, target_len)
        
        loss.backward()
        optim.step()
        
        iterator.set_description(f"epoch{epoch+1} loss:{loss.item()}")

torch.save(model.state_dict(), "CRNN.pth")

model.load_state_dict(torch.load("CRNN.pth", map_location=device))

with torch.no_grad():
    testset = Captcha(pth = "data/CH12", train = False)
    test_img, label = testset[0]
    input_tensor = torch.unsqueeze(torch.tensor(test_img), dim = 0)
    input_tensor = input_tensor.permute(0, 3, 1, 2).to(device)
    
    pred = torch.argmax(model(input_tensor), dim = -1)
    
    prev_letter = pred[0].item()
    pred_word = ""
    for letter in pred:
        if letter.item() != 0 and letter.item() != prev_letter:
            pred_word += list(testset.BOW.keys())[letter.item()]
        prev_letter = letter.item()
    
    plt.imshow(test_img)
    plt.title("prediction: " + pred_word)
    plt.show()