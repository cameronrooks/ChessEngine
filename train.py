import torch
import h5py
import numpy as np
import dataset
import model
import sys
import os


batch_size_ = int(sys.argv[1])
num_epochs = int(sys.argv[2])
model_dir = sys.argv[3]
starting_epoch = 1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set up the model
model = model.Net()

model.to(device)


#if resuming training, load state dict from most recent epoch
weight_files = os.listdir(model_dir)
if(len(weight_files) > 0):

	model_path = ""
	for file_name in weight_files:
    
		index = file_name.find("epoch")
		index = index + 5
		temp = file_name[index:]
        
		if(int(temp) >= starting_epoch):
			starting_epoch = int(temp)
			model_path = file_name

	starting_epoch += 1
	model_path = model_dir + "/" + model_path
	model.load_state_dict(torch.load(model_path))

#load the data
h5f = h5py.File('./data/TrainDataSparse.h5', 'r')

boards = h5f['boards'][:]
labels = h5f['labels'][:]

print(len(np.where(labels < 0)[0]))
print(len(np.where(labels > 0)[0]))

n = boards.shape[0]

h5f.close()

#convert numpy arrays to torch tensors
x_data = torch.from_numpy(boards)
y_data = torch.from_numpy(labels)

x_data = x_data.float()
y_data = y_data.float()


data = dataset.ChessDataset(x_data, y_data)


#create data loaders
train_loader = torch.utils.data.DataLoader(data, batch_size = batch_size_, shuffle = True, num_workers = 0)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.7)
loss_fn = torch.nn.MSELoss()

#training loop
for x in range(num_epochs):
	running_loss = 0
	progress = 0


	for i, data in enumerate(train_loader):

		if (((i * batch_size_) / n) * 100 >= progress):
			print(str(progress) + "%")
			progress += 25

		
		#features and labels
		inputs, labels = data
		inputs, labels = inputs.to(device), labels.to(device)

		#zero gradients
		optimizer.zero_grad()

		#get model predictions
		pred = model(inputs)

		#compute loss and perform backward pass
		loss = loss_fn(pred, labels)
		loss.backward()


		running_loss += loss.item()

		optimizer.step()


	print("epoch " + str(x + starting_epoch) + ": loss = " + str(running_loss))
	torch.save(model.state_dict(), model_dir + "/epoch" + str(starting_epoch + x))

