import torch
import os
import model
import h5py
import dataset

model_path = "./trained_models/model6/epoch113"
batch_size_ = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.Net()
model.to(device)

model.load_state_dict(torch.load(model_path))
model.eval()

h5f = h5py.File('./data/TestData.h5', 'r')

boards = h5f['boards'][:]
labels = h5f['labels'][:]

n = labels.shape[0]

h5f.close()

#convert numpy arrays to torch tensors
x_data = torch.from_numpy(boards)
y_data = torch.from_numpy(labels)

x_data = x_data.float()
y_data = y_data.float()


data = dataset.ChessDataset(x_data, y_data)

test_loader = torch.utils.data.DataLoader(data, batch_size = batch_size_, shuffle = True, num_workers = 0)

loss_fn = torch.nn.MSELoss()

sum = 0

running_loss = 0
for i, data in enumerate(test_loader):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)

    pred = model(inputs)

    loss = loss_fn(pred, labels)
    running_loss += loss.item()


    sum += torch.sum(torch.abs(torch.sub(pred, labels))).item()
    #pred = torch.abs(pred)
    
    #sum += torch.sum(pred)


print(sum/n)

print(running_loss)



    

