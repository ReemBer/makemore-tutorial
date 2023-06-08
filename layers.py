import torch


class Linear:
    def __init__(self, in_features, out_features, bias=True, device='cpu', dtype=torch.float32, gen=None):
        self.W = torch.randn([in_features, out_features], dtype=dtype, device=device, generator=gen)
        self.b = torch.randn(out_features, dtype=dtype, device=device, generator=gen) if bias else None

    def __call__(self, X):
        self.out = X @ self.W
        if self.b is not None:
            self.out += self.b
        return self.out

    def parameters(self):
        return [self.W] + ([] if self.b is None else [self.b])

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def nelement(self):
        return sum(self.parameters())

    def set_bias(self, bias=True, device='cpu', gen=None):
        if self.b is None:
            self.b = torch.randn(out_features, device=device, generator=gen) if bias else None
        else:
            if not bias:
                self.b = None


class LinearNormalized(Linear):
    def __init__(self, in_features, out_features, bias=True, device='cpu', dtype=torch.float32, gen=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype, gen=gen)
        self.W /= in_features ** 0.5


class LinearWithGain(LinearNormalized):
    def __init__(self, in_features, out_features, bias=True, device='cpu', dtype=torch.float32, gen=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype, gen=gen)
        self.W *= 5/3 # a good gain value for tanh() activations


class Tanh():
    def __init__(self, keep_grad=False):
        self.keep_grad = keep_grad

    def __call__(self, X):
        self.out = torch.tanh(X)
        if self.keep_grad:
            self.out.retain_grad()
        return self.out

    def parameters(self):
        return []
