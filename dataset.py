import torch

class ChessDataset(torch.utils.data.Datset):
	def __init__(self, data, labels):
		this.data = data
		this.labels = labels

	def __len__(self):
		return len(this.data)

	def __getitem__(self, idx):
		board = this.data[idx]
		eval = this.labels[idx]

		return board, eval
