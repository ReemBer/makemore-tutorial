import torch
import torch.nn.functional as F

from functools import reduce

from layers import Linear, Tanh


#predicts next token given a sequence of previous ones 
class MultiLayerPerceptron:
    def __init__(self, vocab_size, block_size, emb_size, hidden_size, n_hidden, linear_layer=Linear, keep_intermediate_grad=False, device='cpu', gen=None):
        self.context_size = block_size*emb_size
        self.emb = torch.randn((vocab_size, emb_size), requires_grad = True, device = device)
        self.layers = [linear_layer(self.context_size, hidden_size, device=device, gen=gen), Tanh(keep_intermediate_grad)]
        for i in range(n_hidden-1):
            self.layers.extend([linear_layer(hidden_size, hidden_size, device=device, gen=gen), Tanh(keep_intermediate_grad)])
        self.layers.append(linear_layer(hidden_size, vocab_size, device=device, gen=gen))
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, mini_batch_X):
        embeddings = self.emb[mini_batch_X].view(-1, self.context_size)
        return reduce(lambda out, L: L(out), self.layers, embeddings)

    def parameters(self):
        return [self.emb] + [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, mini_batch_X):
        return self.forward(mini_batch_X)

    def retain_grad(self):
        for p in self.parameters():
            p.retain_grad()

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def nelement(self):
        return sum(p.nelement() for p in self.parameters())
