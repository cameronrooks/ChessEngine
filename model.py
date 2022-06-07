
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(384, 2048)
		self.batch_norm1 = nn.BatchNorm1d(2048)
		self.fc2 = nn.Linear(2048, 2048)
		self.batch_norm2 = nn.BatchNorm1d(2048)
		self.fc3 = nn.Linear(2048, 2048)
		self.batch_norm3 = nn.BatchNorm1d(2048)

		self.out = nn.Linear(2048, 1)

	def forward(self, x):
		x = self.fc1(x)
		x = self.batch_norm1(x)
		x = F.relu(x)
		#x = self.dropout1(x)

		x = self.fc2(x)
		x = self.batch_norm2(x)
		x = F.relu(x)
		#x = self.dropout2(x)

		x = self.fc3(x)
		x = self.batch_norm3(x)
		x = F.relu(x)
		#x = self.dropout3(x)

		x = self.out(x)

		output = x.reshape(-1)

		return output


