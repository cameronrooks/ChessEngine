
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(389, 389)
		self.fc2 = nn.Linear(389, 256)

		self.fc3 = nn.Linear(256, 1)

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)

		output = self.fc3(x)

		output = output.reshape(-1)

		return output


