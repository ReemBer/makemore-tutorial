import torch
import random


class Tokenizer:
    def __init__(self, vocabular):
        self.vocab = vocabular
        self.ctoi = {c:i for i,c in enumerate(vocabular)}
        self.itoc = vocabular

    def encode(self, char_seq):
        return [self.ctoi[c] for c in char_seq]

    def decode(self, token_seq):
        return [self.itoc[t] for t in token_seq]


def build_names_dataset(names, tokenizer, block_size):
    X, y = [], []
    for name in names:
        block = torch.zeros(block_size).to(torch.int64)
        for c in name + '.':
            ix = tokenizer.ctoi[c]
            X.append(block)
            y.append(ix)
            block = block.roll(-1)
            block[-1] = ix
    X, y = torch.stack(X), torch.tensor(y)
    return X, y


class TensorDataset:
    def __init__(self, X, y, device='cpu'):
        self.X = X.to(device)
        self.y = y.to(device)
        self.ids = list(range(X.shape[0]))
        self.l_bord = 0

    def reshuffle(self):
        random.shuffle(self.ids)
        self.l_bord = 0

    def is_processed(self):
        return self.l_bord >= self.X.shape[0]

    def get_mini_batch(self, batch_size, device='cpu'):
        if self.is_processed():
            print("Warning. Current shuffle is processed. Reshuffling...")
            self.reshuffle()
        batch_ids = self.ids[self.l_bord:self.l_bord+batch_size]
        if len(batch_ids) < batch_size:
            k = batch_size - len(batch_ids)
            extra_ids = random.sample(range(self.l_bord), k)
            batch_ids.extend(extra_ids)
        mini_batch_X = self.X[batch_ids].to(device)
        mini_batch_y = self.y[batch_ids].to(device)
        self.l_bord += batch_size
        return mini_batch_X, mini_batch_y