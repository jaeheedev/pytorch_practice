import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
from torch.optim.adam import Adam
import numpy as np
import logging
from torch import optim

# from data import Multi30k
import os
import torch
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T

# from utils import save_pkl, load_pkl
import pickle
import torch
from torchtext.data.metrics import bleu_score


def save_pkl(data, fname):
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def load_pkl(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


def get_bleu_score(output, gt, vocab, specials, max_n=4):

    def itos(x):
        x = list(x.cpu().numpy())
        tokens = vocab.lookup_tokens(x)
        tokens = list(filter(lambda x: x not in {"", " ", "."} and x not in list(specials.keys()), tokens))
        return tokens

    pred = [out.max(dim=1)[1] for out in output]
    pred_str = list(map(itos, pred))
    gt_str = list(map(lambda x: [itos(x)], gt))

    score = bleu_score(pred_str, gt_str, max_n=max_n) * 100
    return  score


def greedy_decode(model, src, max_len, start_symbol, end_symbol):
    src = src.to(model.device)
    src_mask = model.make_src_mask(src).to(model.device)
    memory = model.encode(src, src_mask)

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(model.device)
    for i in range(max_len-1):
        memory = memory.to(model.device)
        tgt_mask = model.make_tgt_mask(ys).to(model.device)
        src_tgt_mask = model.make_src_tgt_mask(src, ys).to(model.device)
        out = model.decode(ys, memory, tgt_mask, src_tgt_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == end_symbol:
            break
    return ys

class Multi30k():

    def __init__(self,
                 lang=("en", "de"),
                 max_seq_len=256,
                 unk_idx=0,
                 pad_idx=1,
                 sos_idx=2,
                 eos_idx=3,
                 vocab_min_freq=2):

        self.dataset_name = "multi30k"
        self.lang_src, self.lang_tgt = lang
        self.max_seq_len = max_seq_len
        self.unk_idx = unk_idx
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.unk = "<unk>"
        self.pad = "<pad>"
        self.sos = "<sos>"
        self.eos = "<eos>"
        self.specials={
                self.unk: self.unk_idx,
                self.pad: self.pad_idx,
                self.sos: self.sos_idx,
                self.eos: self.eos_idx
                }
        self.vocab_min_freq = vocab_min_freq

        self.tokenizer_src = self.build_tokenizer(self.lang_src)
        self.tokenizer_tgt = self.build_tokenizer(self.lang_tgt)

        self.train = None
        self.valid = None
        self.test = None
        self.build_dataset()

        self.vocab_src = None
        self.vocab_tgt = None
        self.build_vocab()

        self.transform_src = None
        self.transform_tgt = None
        self.build_transform()


    def build_dataset(self, raw_dir="raw", cache_dir=".data"):
        cache_dir = os.path.join(cache_dir, self.dataset_name)
        raw_dir = os.path.join(cache_dir, raw_dir)
        os.makedirs(raw_dir, exist_ok=True)

        train_file = os.path.join(cache_dir, "train.pkl")
        valid_file = os.path.join(cache_dir, "valid.pkl")
        test_file = os.path.join(cache_dir, "test.pkl")

        if os.path.exists(train_file):
            self.train = load_pkl(train_file)
        else:
            with open(os.path.join(raw_dir, "train.en"), "r") as f:
                train_en = [text.rstrip() for text in f]
            with open(os.path.join(raw_dir, "train.de"), "r") as f:
                train_de = [text.rstrip() for text in f]
            self.train = [(en, de) for en, de in zip(train_en, train_de)]
            save_pkl(self.train , train_file)

        if os.path.exists(valid_file):
            self.valid = load_pkl(valid_file)
        else:
            with open(os.path.join(raw_dir, "val.en"), "r") as f:
                valid_en = [text.rstrip() for text in f]
            with open(os.path.join(raw_dir, "val.de"), "r") as f:
                valid_de = [text.rstrip() for text in f]
            self.valid = [(en, de) for en, de in zip(valid_en, valid_de)]
            save_pkl(self.valid, valid_file)

        if os.path.exists(test_file):
            self.test = load_pkl(test_file)
        else:
            with open(os.path.join(raw_dir, "test_2016_flickr.en"), "r") as f:
                test_en = [text.rstrip() for text in f]
            with open(os.path.join(raw_dir, "test_2016_flickr.de"), "r") as f:
                test_de = [text.rstrip() for text in f]
            self.test = [(en, de) for en, de in zip(test_en, test_de)]
            save_pkl(self.test, test_file)


    def build_vocab(self, cache_dir=".data"):
        assert self.train is not None
        def yield_tokens(is_src=True):
            for text_pair in self.train:
                if is_src:
                    yield [str(token) for token in self.tokenizer_src(text_pair[0])]
                else:
                    yield [str(token) for token in self.tokenizer_tgt(text_pair[1])]

        cache_dir = os.path.join(cache_dir, self.dataset_name)
        os.makedirs(cache_dir, exist_ok=True)

        vocab_src_file = os.path.join(cache_dir, f"vocab_{self.lang_src}.pkl")
        if os.path.exists(vocab_src_file):
            vocab_src = load_pkl(vocab_src_file)
        else:
            vocab_src = build_vocab_from_iterator(yield_tokens(is_src=True), min_freq=self.vocab_min_freq, specials=self.specials.keys())
            vocab_src.set_default_index(self.unk_idx)
            save_pkl(vocab_src, vocab_src_file)

        vocab_tgt_file = os.path.join(cache_dir, f"vocab_{self.lang_tgt}.pkl")
        if os.path.exists(vocab_tgt_file):
            vocab_tgt = load_pkl(vocab_tgt_file)
        else:
            vocab_tgt = build_vocab_from_iterator(yield_tokens(is_src=False), min_freq=self.vocab_min_freq, specials=self.specials.keys())
            vocab_tgt.set_default_index(self.unk_idx)
            save_pkl(vocab_tgt, vocab_tgt_file)

        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt


    def build_tokenizer(self, lang):
        from torchtext.data.utils import get_tokenizer
        spacy_lang_dict = {
                'en': "en_core_web_sm",
                'de': "de_core_news_sm"
                }
        assert lang in spacy_lang_dict.keys()
        return get_tokenizer("spacy", spacy_lang_dict[lang])


    def build_transform(self):
        def get_transform(self, vocab):
            return T.Sequential(
                    T.VocabTransform(vocab),
                    T.Truncate(self.max_seq_len-2),
                    T.AddToken(token=self.sos_idx, begin=True),
                    T.AddToken(token=self.eos_idx, begin=False),
                    T.ToTensor(padding_value=self.pad_idx))

        self.transform_src = get_transform(self, self.vocab_src)
        self.transform_tgt = get_transform(self, self.vocab_tgt)


    def collate_fn(self, pairs):
        src = [self.tokenizer_src(pair[0]) for pair in pairs]
        tgt = [self.tokenizer_tgt(pair[1]) for pair in pairs]
        batch_src = self.transform_src(src)
        batch_tgt = self.transform_tgt(tgt)
        return (batch_src, batch_tgt)


    def get_iter(self, **kwargs):
        if self.transform_src is None:
            self.build_transform()
        train_iter = DataLoader(self.train, collate_fn=self.collate_fn, **kwargs)
        valid_iter = DataLoader(self.valid, collate_fn=self.collate_fn, **kwargs)
        test_iter = DataLoader(self.test, collate_fn=self.collate_fn, **kwargs)
        return train_iter, valid_iter, test_iter


    def translate(self, model, src_sentence: str, decode_func):
        model.eval()
        src = self.transform_src([self.tokenizer_src(src_sentence)]).view(1, -1)
        num_tokens = src.shape[1]
        tgt_tokens = decode_func(model,
                                 src,
                                 max_len=num_tokens+5,
                                 start_symbol=self.sos_idx,
                                 end_symbol=self.eos_idx).flatten().cpu().numpy()
        tgt_sentence = " ".join(self.vocab_tgt.lookup_tokens(tgt_tokens))
        return tgt_sentence

class Transformer(nn.Module):
    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def encode(self, src, src_mask):
        out = self.encode(self.src_embed(src), src_mask)
        return out
    
    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = self.decode(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)
        return out

    def forward(self, src, tgt):
        src_mask = self.maks_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1)
        return out, decoder_out

    def make_pad_mask(self, query, key, pad_idx=1):
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask

    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask

    def make_subsequent_mask(query, key):
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8')
        mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)
        return mask

    def make_tgt_mask(self, tgt):
        pad_mask = self.make_pad_mask(tgt, tgt)
        seq_mask = self.make_subsequent_mask(tgt, tgt)
        mask = pad_mask & seq_mask
        return pad_mask & seq_mask

    def make_src_tgt_mask(self, src, tgt):
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask

class Encoder(nn.Module):
    def __init__(self, encoder_block, n_layer):
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_block))
        
    def forward(self, src, src_mask):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, decoder_block, n_layer):
        super(Decoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.n_layer)])

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        for layer in self.layers:
            out = layer(out, encoder_out, tgt_mask, src_tgt_mask)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(2)]
    
    def forward(self, src, src_mask):
        out = src
        out = self.residuals[0](out, lambda out: self.self_attention(query=out, key=out, value=out, mask=src_mask))
        out = self.residuals[1](out, self.position_ff)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, self_attention, cross_attention, position_ff):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attentnion = cross_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(3)]

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out= tgt
        out = self.residuals[0](out, lambda out: self.self_attention(query=out, key=out, value=out, mask=tgt_mask))
        out = self.residuals[1](out, lambda out: self.cross_attentnion(query=out, key = encoder_out, value = src_tgt_mask))
        out = self.residuals[2](out, self.position_ff)
        return out

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, h, qkv_fc, out_fc):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = copy.deepcopy(qkv_fc)
        self.k_fc = copy.deepcopy(qkv_fc)
        self.v_fc = copy.deepcopy(qkv_fc)
        self.out_fc = out_fc
    
    def calculate_attention(self, query, key, value, mask):
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1))
        attention_score = attention_score / math.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        attention_prob = F.softmax(attention_score, dim=-1)
        out = torch.matmul(attention_prob, value)
        return out

    def forward(self, query, key, value, mask =None):
        n_batch = query.size(0)

        def transform(x, fc):
            out = fc(x)
            out = out.view(n_batch, -1, self.h, self.d_model // self.h)
            out = out.transpose(1, 2)
            return out

        query = transform(query, self.q_fc)
        key = transform(key, self.k_fc)
        value = transform(value, self.v_fc)

        out = self.calculate_attention(query, key, value, mask)
        out = out.transpose(1, 2)
        out = out.contiguous().view(n_batch, -1, self.d_model)
        out = self.out_fc(out)
        return out

class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, fc1, fc2):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1
        self.relu = nn.ReLU()
        self.fc2 = fc2

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out= self.relu(out)
        out = self.fc2(out)
        return out

class ResidualConnectionLayer(nn.Module):
    def __init__(self):
        super(ResidualConnectionLayer, self).__init__()

    def forward(self, x, sub_layer):
        out = x
        out = sub_layer(out)
        out = out + x
        return out

class TransformerEmbedding(nn.Module):
    def __init__(self, token_embed, pos_embed):
        super(TransformerEmbedding, self).__init__()
        self.embedding = nn.Sequential(token_embed, pos_embed)

    def forward(self, x):
        out = self.embedding(x)
        return out

class TokenEmbedding(nn.Module):
    def __init__(self, d_embed, vocab_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.d_embed = d_embed
    
    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_embed)
        return out

class PositionalEncoding(nn.Module): # normalizing positonal information
    def __init__(self, d_embed, max_len=256, device=torch.device("cpu")):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeors(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0)/d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0).to(device)

    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out

def build_model(src_vocab_size, tgt_vocab_size, device = torch.device("cpu"), max_len=256, d_embed=512, n_layer = 6, d_model = 512, h=8, d_ff=2048):
    import copy
    copy = copy.deepcopy

    src_token_embed = TokenEmbedding(d_embed=d_embed, vocab_size=src_vocab_size)
    tgt_token_embed = TokenEmbedding(d_embed=d_embed, vocab_size=tgt_vocab_size)
    
    pos_embed = PositionalEncoding(d_embed=d_embed, max_len=max_len, device=device)
    
    src_embed = TransformerEmbedding(token_embed=src_token_embed, pos_embed=copy(pos_embed))
    tgt_embed = TransformerEmbedding(token_embed=tgt_token_embed, pos_embed=copy(pos_embed))

    attention = MultiHeadAttentionLayer(d_model=d_model, h=h, qkv_fc=nn.Linear(d_embed, d_model), out_fc=nn.Linear(d_model, d_embed))
    
    position_ff = PositionWiseFeedForwardLayer(fc1=nn.Linear(d_embed, d_ff), fc2=nn.Linear(d_ff, d_embed))

    encoder_block = EncoderBlock(self_attention=copy(attention), position_ff=copy(position_ff))
    decoder_block = DecoderBlock(self_attention=copy(attention), cross_attention=copy(attention), position_ff=copy(position_ff))

    encoder = Encoder(encoder_block=encoder_block, decoder_block=decoder_block)
    decoder = Decoder(decoder_block=decoder_block, n_layer=n_layer)

    generator = nn.Linear(d_model, tgt_vocab_size)

    model = Transformer(src_embed=src_embed, tgt_embed=tgt_embed, encoder=encoder, decoder=decoder, generator=generator).to(device)
    
    model.device = device
    
    return model

DEVICE = torch.device('cuda:0')
CHECKPOINT_DIR = "./checkpoint"

N_EPOCH = 1000

BATCH_SIZE = 2048
NUM_WORKERS = 8

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-4
ADAM_EPS = 5e-9
SCHEDULER_FACTOR = 0.9
SCHEDULER_PATIENCE = 10

WARM_UP_STEP = 100

DROPOUT_RATE = 0.1


DATASET = Multi30k()


def train(model, data_loader, optimizer, criterion, epoch, checkpoint_dir):
    model.train()
    epoch_loss = 0

    for idx, (src, tgt) in enumerate(data_loader):
        src = src.to(model.device)
        tgt = tgt.to(model.device)
        tgt_x = tgt[:, :-1]
        tgt_y = tgt[:, 1:]

        optimizer.zero_grad()

        output, _ = model(src, tgt_x)

        y_hat = output.contiguous().view(-1, output.shape[-1])
        y_gt = tgt_y.contiguous().view(-1)
        loss = criterion(y_hat, y_gt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
    num_samples = idx + 1

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f"{epoch:04d}.pt")
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                   }, checkpoint_file)

    return epoch_loss / num_samples


def evaluate(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0

    total_bleu = []
    with torch.no_grad():
        for idx, (src, tgt) in enumerate(data_loader):
            src = src.to(model.device)
            tgt = tgt.to(model.device)
            tgt_x = tgt[:, :-1]
            tgt_y = tgt[:, 1:]

            output, _ = model(src, tgt_x)

            y_hat = output.contiguous().view(-1, output.shape[-1])
            y_gt = tgt_y.contiguous().view(-1)
            loss = criterion(y_hat, y_gt)

            epoch_loss += loss.item()
            score = get_bleu_score(output, tgt_y, DATASET.vocab_tgt, DATASET.specials)
            total_bleu.append(score)
        num_samples = idx + 1

    loss_avr = epoch_loss / num_samples
    bleu_score = sum(total_bleu) / len(total_bleu)
    return loss_avr, bleu_score


def main():
    model = build_model(len(DATASET.vocab_src), len(DATASET.vocab_tgt), device=DEVICE, dr_rate=DROPOUT_RATE)

    def initialize_weights(model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.kaiming_uniform_(model.weight.data)

    model.apply(initialize_weights)

    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

    criterion = nn.CrossEntropyLoss(ignore_index=DATASET.pad_idx)

    train_iter, valid_iter, test_iter = DATASET.get_iter(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    for epoch in range(N_EPOCH):
        logging.info(f"*****epoch: {epoch:02}*****")
        train_loss = train(model, train_iter, optimizer, criterion, epoch, CHECKPOINT_DIR)
        logging.info(f"train_loss: {train_loss:.5f}")
        valid_loss, bleu_score  = evaluate(model, valid_iter, criterion)
        if epoch > WARM_UP_STEP:
            scheduler.step(valid_loss)
        logging.info(f"valid_loss: {valid_loss:.5f}, bleu_score: {bleu_score:.5f}")

        logging.info(DATASET.translate(model, "A little girl climbing into a wooden playhouse .", greedy_decode))
        # expected output: "Ein kleines Mädchen klettert in ein Spielhaus aus Holz ."

    test_loss, bleu_score = evaluate(model, test_iter, criterion)
    logging.info(f"test_loss: {test_loss:.5f}, bleu_score: {bleu_score:.5f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    logging.basicConfig(level=logging.INFO)
    main()  

