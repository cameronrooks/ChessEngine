from test import test_
import model
import h5py
import os
import sys
import dataset
import torch
import matplotlib as mp

model_dir = './trained_models/model10'
weight_files = os.listdir(model_dir)

test_losses = [] * len(model_dir)
train_losses = [] * len(model_dir)
#weight_files.sort()

print(weight_files)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.Net()

model.to(device)

h5f_train = h5py.File('./data/TrainDataSparse.h5', 'r')
h5f_test = h5py.File('./data/TestDataSparse.h5', 'r')

boards_train = h5f_train['boards'][:]
labels_train = h5f_train['labels'][:]

boards_test = h5f_test['boards'][:]
labels_test = h5f_test['labels'][:]

#convert numpy arrays to torch tensors
x_train = torch.from_numpy(boards_train)
y_train = torch.from_numpy(labels_train)

x_train = x_train.float()
y_train = y_train.float()

x_test = torch.from_numpy(boards_test)
y_test = torch.from_numpy(labels_test)

x_test = x_test.float()
y_test = y_test.float()


train_data = dataset.ChessDataset(x_train, y_train)
test_data = dataset.ChessDataset(x_test, y_test)

train_loader = torch.utils.data.DataLoader(train_data, batch_size = 256, shuffle = True, num_workers = 0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 256, shuffle = True, num_workers = 0)


count = 0
for epoch in weight_files:
    index = epoch.find("epoch")
    index = index + 5
    temp = epoch[index:]

    epoch_number = int(temp)

    epoch_file = model_dir + '/' + epoch

    test_loss = test_(model, epoch_file, test_loader, device)
    train_loss = test_(model, epoch_file, train_loader, device)

    test_losses[epoch_number - 1] = test_loss
    train_losses[epoch_number - 1] = train_loss

    print(count, flush = True)
    count += 1


x_vals = range(1, len(model_dir) + 1)
mp.pyplot.plot(xvals, test_losses, color = 'blue')
mp.pyplot.plot(xvals, train_losses, color = 'orange')