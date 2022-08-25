from utils import test_
import model
import h5py
import os
import sys
import dataset
import torch
import matplotlib.pyplot as plt

model_dir = './trained_models/model1'
weight_files = os.listdir(model_dir + '/epochs')



test_losses = [0] * len(weight_files)
train_losses = []
#weight_files.sort()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.Net()

model.to(device)

h5f_test = h5py.File('./data/TestDataSparse.h5', 'r')

boards_test = h5f_test['boards'][:]
labels_test = h5f_test['labels'][:]

#convert numpy arrays to torch tensors

x_test = torch.from_numpy(boards_test)
y_test = torch.from_numpy(labels_test)

x_test = x_test.float()
y_test = y_test.float()


test_data = dataset.ChessDataset(x_test, y_test)

test_loader = torch.utils.data.DataLoader(test_data, batch_size = 512, shuffle = True, num_workers = 0)


#get training losses from text file
f = open(model_dir + "/train_losses.txt", 'r')
lines = f.readlines()

for line in lines:
    train_losses.append(float(line))

#iterate through every epoch and calculate test loss

count = 0
for epoch in weight_files:
    #get epoch number for current file
    index = epoch.find("epoch")
    index = index + 5
    temp = epoch[index:]

    epoch_number = int(temp)

    epoch_file = model_dir + '/epochs/' + epoch

    #calculate loss
    test_loss = test_(model, epoch_file, test_loader, device)

    #enter loss in array at correct position
    test_losses[epoch_number - 1] = test_loss

    print(count, flush = True)
    count += 1



x_vals = range(1, len(weight_files) + 1)

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

ax1.plot(x_vals, test_losses)
ax1.set_title("Test Loss Vs Epoch")
ax1.set_xlabel("Epoch Number")

ax2.plot(x_vals, train_losses)
ax2.set_title("Train Loss Vs Epoch")
ax2.set_xlabel("Epoch Number")

ax3.plot(x_vals, test_losses, color = "blue", label = "test loss")
ax3.plot(x_vals, train_losses, color = "orange", label = "train loss")
ax3.set_xlabel("Epoch Number")


plt.show()
#mp.plot(x_vals, test_losses, color = 'blue')
#mp.show()
#mp.pyplot.plot(xvals, train_losses, color = 'orange')
