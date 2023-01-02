import string
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import random
import tqdm
from torch.optim.adam import Adam

def get_BOW(corpus):
    BOW = {"<SOS>":0, "<EOS>":1}
    
    for line in corpus:
        for word in line.split():
            if word not in BOW.keys():
                BOW[word] = len(BOW.keys())
                
    return BOW

class Eng2Kor(Dataset):
    def __init__(self, pth2txt = "data/CH11.txt"):
        self.eng_corpus = []
        self.kor_corpus = []
        
        with open(pth2txt, 'r', encoding="utf-8") as f:
            lines = f.read().split("\n")
            for line in lines:
                txt = "".join(v for v in line if v not in string.punctuation).lower()
                engtxt = txt.split("\t")[0]
                kortxt = txt.split("\t")[1]
                
                if len(engtxt.split()) <= 10 and len(kortxt.split()) <= 10:
                    self.eng_corpus.append(engtxt)
                    self.kor_corpus.append(kortxt)
                    
        self.engBOW = get_BOW(self.eng_corpus)
        self.korBOW = get_BOW(self.kor_corpus)
        
    def gen_seq(self,line):
        seq = line.split()
        seq.append("<EOS>")
        
        return seq
    
    def __len__(self):
        return len(self.eng_corpus)
    
    def __getitem__(self, i):
        data = np.array([self.engBOW[txt] for txt in self.gen_seq(self.eng_corpus[i])])
        label = np.array([self.korBOW[txt] for txt in self.gen_seq(self.kor_corpus[i])])
        
        return data, label
    
def loader(dataset):
    for i in range(len(dataset)):
        data, label = dataset[i]
        yield torch.tensor(data), torch.tensor(label)
            
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        
    def forward(self, x, h):
        x = self.embedding(x).view(1, 1, -1)
        output, hidden = self.gru(x, h)
        return output, hidden
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p = 0.1, max_length=11):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        
        self.attention = nn.Linear(hidden_size * 2, max_length)
        
        self.context = nn.Linear(hidden_size * 2, hidden_size)
        
        self.dropout = nn.Dropout(dropout_p)
        
        self.gru = nn.GRU(hidden_size, hidden_size)
        
        self.out = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, h, encoder_outputs):
        x = self.embedding(x).view(1, 1, -1)
        x = self.dropout(x)
        
        attn_weights = self.softmax(self.attention(torch.cat((x[0], h[0]), -1)))
        
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        
        output = torch.cat((x[0], attn_applied[0]), 1)
        output = self.context(output).unsqueeze(0)
        output = self.relu(output)
        
        output, hidden = self.gru(output, h)
        
        output = self.out(output[0])
        
        return output
    
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = Eng2Kor()

encoder = Encoder(input_size=len(dataset.engBOW), hidden_size=64).to(device)
decoder = Decoder(64, len(dataset.korBOW), dropout_p=0.1).to(device)

encoder_optimizer = Adam(encoder.parameters(), lr = 0.0001)
decoder_optimizer = Adam(decoder.parameters(), lr = 0.0001)

for epoch in range(200):
    iterator = tqdm.tqdm(loader(dataset), total=len(dataset))
    total_loss = 0
    
    for data, label in iterator:
        data = torch.tensor(data, dtype = torch.long).to(device)
        label = torch.tensor(label, dtype = torch.long).to(device)
        
        encoder_hidden = torch.zeros(1, 1, 64).to(device)
        encoder_outputs = torch.zeros(11, 64).to(device)
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        loss = 0
        
        for ei in range(len(data)):
            encoder_output, encoder_hidden = encoder(data[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0,0]
            
        decoder_input = torch.tensor([[0]]).to(device)
        
        decoder_hidden = encoder_hidden
        
        use_teacher_forcing = True if random.random() <0.5 else False
        
        if use_teacher_forcing:
            for di in range(len(label)):
                decoder_output = decoder(decoder_input, decoder_hidden, encoder_outputs)
                target = torch.tensor(label[di], dtype = torch.long).to(device)
                target = torch.unsqueeze(target, dim=0).to(device)
                loss += nn.CrossEntropyLoss()(decoder_output, target)
                decoder_input = target
        else:
            for di in range(len(label)):
                decoder_output = decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                
                target = torch.tensor(label[di], dtype = torch.long).to(device)
                target = torch.unsqueeze(target, dim=0).to(device)
                loss += nn.CrossEntropyLoss()(decoder_output, target)
                
                if decoder_input.item() == 1:
                    break
        total_loss += loss.item()/len(dataset)
        iterator.set_description(f"epoch:{epoch+1} loss:{total_loss}")
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        
torch.save(encoder.state_dict(), 'attn_enc.pth')
torch.save(decoder.state_dict(), 'attn_dec.pth')\


encoder.load_state_dict(torch.load("attn_enc.pth", map_location=device))
decoder.load_state_dict(torch.load("attn_dec.pth", map_location=device))

idx = random.randint(0, len(dataset))

input_sentence = dataset.eng_corpus[idx]
pred_sentence = ""

data, label = dataset[idx]
data = torch.tensor(data, dtype=torch.long).to(device)
label = torch.tensor(data, dtype = torch.long).to(device)

encoder_hidden = torch.zeros(1, 1, 64).to(device)
encoder_outputs = torch.zeros(11, 64).to(device)

for ei in range(len(data)):
    encoder_output, encoder_hidden = encoder(data[ei], encoder_hidden)
    encoder_outputs[ei] = encoder_output[0, 0]
    
decoder_input = torch.tensor([[0]]).to(device)

decoder_hidden = encoder_hidden

for di in range(11):
    decoder_output = decoder(decoder_input, decoder_hidden, encoder_outputs)
    topv, topi = decoder_output.topk(1)
    decoder_input = topi.squeeze().detach()
    
    if decoder_input.item() == 1:
        break
    pred_sentence += list(dataset.korBOW.keys())[decoder_input] + " "

print(input_sentence)
print(pred_sentence)