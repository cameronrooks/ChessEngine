import torch
import os
import model
import h5py
import dataset


def test_(model, model_path, data_loader, device):
    batch_size_ = 256

    model.load_state_dict(torch.load(model_path))
    model.eval()

    loss_fn = torch.nn.MSELoss()

    sum = 0

    running_loss = 0
    for i, data in enumerate(data_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        pred = model(inputs)

        loss = loss_fn(pred, labels)
        running_loss += loss.item()


        sum += torch.sum(torch.abs(torch.sub(pred, labels))).item()

    return running_loss



    

