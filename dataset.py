import torch

class ChessDataset(torch.utils.data.Dataset):
	def __init__(self, data, labels):
		self.data = data
		self.labels = labels

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		board = self.data[idx]
		eval = self.labels[idx]

		return board, eval
