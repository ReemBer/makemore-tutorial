import torch
import numpy as np

import random

from dataset import Tokenizer, build_names_dataset, TensorDataset
from lr_scheduler import ConstatntLr
from optim import SgdOptimizer
from train import train
from model import MlpWithBatchNorm

import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# read names
with open("names.txt", "r") as fnames:
    names = fnames.read().splitlines()
names_cnt = len(names)

special_tokens = ['.']
vocab = special_tokens + sorted(list(set(''.join(names))))
vocab_size = len(vocab)
tokenizer = Tokenizer(vocab)

block_size = 3 # represents the length of context
random.shuffle(names)
train_bound = int(0.8*names_cnt)
val_bound   = int(0.9*names_cnt)
train_X, train_y = build_names_dataset(names[:train_bound], tokenizer, block_size)
val_X, val_y = build_names_dataset(names[train_bound:val_bound], tokenizer, block_size)
test_X, test_y = build_names_dataset(names[val_bound:], tokenizer, block_size)
train_ds = TensorDataset(train_X, train_y)
val_ds   = TensorDataset(val_X, val_y, device)

fig, ax = plt.subplots()
scatter = ax.scatter([], [], s=200)
text = None

def update(step, C):
    global scatter
    global text
    scatter.remove()
    scatter = ax.scatter(C[:,0].data, C[:,1].data, color='blue', s=200)
    x_l = C[:,0].data.min() * 1.15
    x_r = C[:,0].data.max() * 1.15
    y_l = C[:,1].data.min() * 1.15
    y_r = C[:,1].data.max() * 1.15
    first_text = False
    if text is None:
        text = [0] * C.shape[0]
        first_text = True
    for i in range(C.shape[0]):
        if not first_text:
            text[i].remove()
        text[i] = ax.text(C[i,0].item(), C[i,1].item(), tokenizer.itoc[i], ha="center", va="center", color='white')
    ax.set_xlim(x_l, x_r)
    ax.set_ylim(y_l, y_r)
    ax.set_title(f'step {step}')

def print_embeddings(step, model):
    update(step, model.emb.cpu())
    plt.pause(0.001)

# animation = FuncAnimation(fig, update, frames=1002, interval=80, blit=False)
plt.grid('minor')
plt.ion()
plt.show()


g = torch.Generator(device=device).manual_seed(7877)
model = MlpWithBatchNorm(vocab_size, block_size, emb_size=2, hidden_size=128, n_hidden=4, device=device, gen=g)
lr_provider = ConstatntLr(0.07)
optimizer   = SgdOptimizer(model.parameters(), lr_provider)

print("start training ...")
t_loss, v_loss = train(
    model,
    train_ds,
    val_ds,
    epoches=3,
    batch_size=192,
    optimizer=optimizer,
    step_callback=print_embeddings,
    device=device
)