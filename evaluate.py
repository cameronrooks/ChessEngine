import model
import chess
import torch
from utils import encode_board, get_eval_from_model_output, encode_board_without_turnbit

model_path = "./trained_models/model2/epochs/epoch104"

fen_str = "rnb2bnr/pppppkpp/8/2Q5/8/8/PPPPPPPP/RNB1K1NR b KQ - 0 1"

model = model.Net()
model.load_state_dict(torch.load(model_path))
model.eval()

#encoded_board = encode_board(fen_str)
encoded_board = encode_board_without_turnbit(fen_str)

#print(encoded_board)

encoded_board = torch.from_numpy(encoded_board).float()
encoded_board = encoded_board.unsqueeze(0)


eval_pred = model(encoded_board)
eval_ = get_eval_from_model_output(eval_pred.item())

print(eval_)
