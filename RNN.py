import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn
import tqdm
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader

# prepare dataset
class Netflix(Dataset):
    def __init__(self):
        data = pd.read_csv("data/CH06.csv")
        self.csv = pd.read_csv("data/CH06.csv")
        
        # input data
        self.data = self.csv.iloc[:, 1:4].values
        self.data = self.data / np.max(self.data) # normalization
        
        # target data
        self.label = data["Close"].values
        self.label = self.label / np.max(self.label)
        
    def __len__(self):
        # usable batch number
        return len(self.data) - 30
    
    def __getitem__(self, i):
        data = self.data[i:i+30] # 30 of input data
        label = self.label[i+30] # 30 of target data
        
        return data, label
    
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        
        self.rnn = nn.RNN(input_size = 3, hidden_size = 8, num_layers = 5, batch_first = True)
        
        self.fc1 = nn.Linear(in_features = 240, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x, h0):
        x, hn = self.rnn(x, h0)
        
        # removing time sequence
        x = torch.reshape(x, (x.shape[0], -1))
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        x = torch.flatten(x)
        
        return x


# train
device = "cuda" if torch.cuda.is_available() else "cpu"

model = RNN().to(device)
dataset = Netflix()

loader = DataLoader(dataset, batch_size=32)
optim = Adam(params = model.parameters(), lr = 0.0001)

for epoch in range(200):
    iterator = tqdm.tqdm(loader)
    for data, label in iterator:
        optim.zero_grad()
        
        h0 = torch.zeros(5, data.shape[0], 8).to(device)
        
        pred = model(data.type(torch.FloatTensor).to(device), h0)
        
        loss = nn.MSELoss()(pred, label.type(torch.FloatTensor).to(device))
        loss.backward()
        optim.step()
        
        iterator.set_description(f"epoch{epoch} loss:{loss.item()}")
        
torch.save(model.state_dict(), "./rnn.pth") # saving model

# test
loader = DataLoader(dataset, batch_size=1)

preds = []
total_loss = 0

with torch.no_grad():
    model.load_state_dict(torch.load("rnn.pth", map_location=device))
    
    for data, label in loader:
        h0 = torch.zeros(5, data.shape[0], 8).to(device)
        
        pred = model(data.type(torch.FloatTensor).to(device), h0)
        preds.append(pred.item())
        
        loss = nn.MSELoss()(pred, label.type(torch.FloatTensor).to(device))
        total_loss += loss/len(loader)
total_loss.item()

plt.plot(preds, label = "prediction")
plt.plot(dataset.label[30:], label = "actual")
plt.legend()
plt.show()