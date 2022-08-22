import sys
sys.path.append('..')

import torch
import model
from utils import encode_board, get_eval_from_model_output
import chess
from operator import itemgetter

class ChessEngine():
    def __init__(self, model_path, max_depth = 4):
        self.max_depth = max_depth
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #set up the model
        self.model = model.Net()
        self.model.eval()

        #self.model.to(device)

        self.model.load_state_dict(torch.load(model_path))

    def minimax(self, maximizing, chess_board, depth):
        fen_str = chess_board.fen()

        if (chess_board.is_checkmate()):
            if (" w " in fen_str):
                return -100
            else:
                return 100

        encoded_board = encode_board(fen_str)

        encoded_board = torch.from_numpy(encoded_board).float()
        encoded_board = encoded_board.unsqueeze(0)

        eval_pred = self.model(encoded_board)
        eval_ = get_eval_from_model_output(eval_pred.item())

        if (depth == self.max_depth):
            return eval_

        scores = []

        next_depth = depth + 1
        legal_moves = list(chess_board.legal_moves)
        for move in legal_moves:
            chess_board.push(move)
            scores.append(self.minimax(not maximizing, chess_board, next_depth))
            #print(scores)
           # print("\n\n")
            chess_board.pop()


        #print(scores)
        max_ = max(scores)
        min_ = min(scores)

        #print(scores)
        if (maximizing):
            if depth > 0:
                return max_
            else:
                max_index = scores.index(max_)
                return max_, legal_moves[max_index]

        else:
            if depth > 0:
                return min_
            else:
                min_index = scores.index(min_)
                return min_, legal_moves[min_index]

    def get_best_move(self, chess_board):
        fen_str = chess_board.fen()
        if (" w " in fen_str):
            maximizing = False
        else:
            maximizing = True

        eval_, move = self.minimax(not maximizing, chess_board, 0)

        return eval_, move

