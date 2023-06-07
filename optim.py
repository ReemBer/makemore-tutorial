import torch

class SgdOptimizer:
    def __init__(self, parameters, learning_rate_provider):
        self.parameters = parameters
        self.lr_provider = learning_rate_provider
        self.update_data_ratio = []

    def step(self):
        lr = self.lr_provider.get()
        for p in self.parameters:
            p.data -= lr*p.grad
        with torch.no_grad():
            self.update_data_ratio.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in self.parameters])
