import torch
import h5py
import numpy as np
import dataset
import model
import sys


batch_size_ = 10
num_epochs = 25

#load the preprocessed data
h5f_boards = h5py.File('./data/ChessData.h5', 'r')
h5f_labels = h5py.File('./data/ChessLabels.h5', 'r')

boards = h5f_boards['boards'][:]
labels = h5f_labels['labels'][:]

h5f_boards.close()
h5f_labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#convert numpy arrays to torch tensors
x_data = torch.from_numpy(boards)
y_data = torch.from_numpy(labels)

x_data = x_data.float()
y_data = y_data.float()

sys.stdout.flush()

train_size = int(0.8 * len(x_data))
test_size = len(x_data) - train_size


data = dataset.ChessDataset(x_data, y_data)


#split data into train and test
train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

#create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size_, shuffle = True, num_workers = 0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size_, shuffle = True, num_workers = 0)


#set up the model
model = model.Net()
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
loss_fn = torch.nn.MSELoss()

#training loop
for x in range(num_epochs):
	running_loss = 0


	for i, data in enumerate(train_loader):

		if ((i * batch_size_) % 250000 == 0):
			print(i*batch_size_)
			sys.stdout.flush()

		inputs, labels = data

		inputs, labels = inputs.to(device), labels.to(device)


		#print(inputs.shape)
		#print(labels.shape)

		optimizer.zero_grad()

		pred = model(inputs)

		sys.stdout.flush()

		loss = loss_fn(pred, labels)
		loss.backward()


		running_loss += loss.item()

		optimizer.step()


	print("epoch" + str(x) + ": loss = " + str(running_loss))


torch.save(model.state_dict(), "./trained_models/model")
