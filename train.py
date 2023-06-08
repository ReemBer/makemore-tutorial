import torch
import torch.nn.functional as F


def train_step(model, mini_batch_X, mini_batch_y, optimizer):
    logits = model(mini_batch_X)
    loss = F.cross_entropy(logits, mini_batch_y)
    model.retain_grad()
    model.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def train(model, train_dataset, val_dataset, epoches, batch_size, optimizer, device='cpu'):
    train_loss, val_loss = [], []
    for epoch in range(epoches):
        train_dataset.reshuffle()
        cur_losses = []
        while not train_dataset.is_processed():
            mini_batch_X, mini_batch_y = train_dataset.get_mini_batch(batch_size, device)
            loss = train_step(model, mini_batch_X, mini_batch_y, optimizer)
            cur_losses.append(loss.item())
            mini_batch_X.detach().cpu()
            mini_batch_y.detach().cpu()
        cur_avg_train_loss = sum(cur_losses) / len(cur_losses)
        cur_val_loss = F.cross_entropy(model(val_dataset.X), val_dataset.y).item()
        print(f'{epoch=}: {cur_avg_train_loss=}, {cur_val_loss=}')
        train_loss.append(cur_avg_train_loss)
        val_loss.append(cur_val_loss)
    return train_loss, val_loss
