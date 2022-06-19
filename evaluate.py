import model
import chess
import torch
from utils import encode_board

model_path = "./trained_models/model7/epoch95"

fen_str = "3r1rk1/ppp1qppp/2n1n3/2b1pP2/3p4/1B1P1N2/PPPN1PPP/R2QKB1R b KQ - 0 1"

model = model.Net()
model.load_state_dict(torch.load(model_path))
model.eval()

encoded_board = encode_board(fen_str)

#print(encoded_board)

encoded_board = torch.from_numpy(encoded_board).float()
encoded_board = encoded_board.unsqueeze(0)


eval_ = model(encoded_board)

print(eval_.item())