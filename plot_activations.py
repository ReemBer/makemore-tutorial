import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from layers import Tanh
from train import train_step, train

def plot_forward_activations(model, layer_type=None):
    plt.figure(figsize=(20, 4))
    legends = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, layer_type):
            t = layer.out.to('cpu')
            saturation = (t.abs() > 0.97).float().mean()*100
            layer_name = layer.__class__.__name__
            print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer_name, t.mean(), t.std(), saturation))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'layer {i} ({layer.__class__.__name__}')
    plt.legend(legends);
    plt.title('activation distribution')


def plot_backward_gradients(model, layer_type=None):
    plt.figure(figsize=(20, 4))
    legends = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, layer_type):
            t = layer.out.grad.to('cpu')
            layer_name = layer.__class__.__name__
            print('layer %d (%10s): mean %+f, std %e' % (i, layer_name, t.mean(), t.std()))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'layer {i} ({layer.__class__.__name__}')
    plt.legend(legends);
    plt.title('gradient distribution')


def plot_weights_gradients(model):
    plt.figure(figsize=(20, 4))
    legends = []
    for i,p in enumerate(model.parameters()):
        t = p.grad.to('cpu')
        if p.ndim == 2:
            print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'{i} {tuple(p.shape)}')
    plt.legend(legends)
    plt.title('weights gradient distribution');


def plot_gradient_to_weight_ratio(optimizer):
    plt.figure(figsize=(20, 4))
    legends = []
    ud = optimizer.update_data_ratio
    for i,p in enumerate(optimizer.parameters):
        if p.ndim == 2:
            plt.plot([ud[j][i] for j in range(len(ud))])
            legends.append('param %d' % i)
    plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
    plt.legend(legends);

def plot_initialization_statistics(model, optimizer, train_ds, val_ds, device='cpu'):
    init_t_loss = F.cross_entropy(model(train_ds.X.to(device)), train_ds.y.to(device)).item()
    init_v_loss = F.cross_entropy(model(val_ds.X.to(device)), val_ds. y.to(device)).item()
    print(f"{init_t_loss=}, {init_v_loss=}")
    plot_forward_activations(model, layer_type=Tanh)
    mb_X, mb_y = train_ds.get_mini_batch(64, device)
    train_step(model, mb_X, mb_y, optimizer)
    plot_backward_gradients(model, Tanh)
    plot_weights_gradients(model)
    t_loss, v_loss = train(model, train_ds, val_ds, epoches=1, batch_size=182, optimizer=optimizer, device=device)
    plot_gradient_to_weight_ratio(optimizer)
