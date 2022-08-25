import sys
sys.path.append('..')

import torch
import model
from utils import encode_board, get_eval_from_model_output, encode_board_without_turnbit
import chess
from operator import itemgetter

class ChessEngine():
    def __init__(self, model_path, max_depth = 3):
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

    def minimax_with_ab_pruning(self, maximizing, board, depth, alpha, beta):
        fen_str = board.fen()

        if (board.is_checkmate()):
            if (" w " in fen_str):
                return -100 + depth
            else:
                return 100 - depth


        if (depth == self.max_depth):
            encoded_board = encode_board(fen_str)

            encoded_board = torch.from_numpy(encoded_board).float()
            encoded_board = encoded_board.unsqueeze(0)

            eval_pred = self.model(encoded_board)
            eval_ = get_eval_from_model_output(eval_pred.item())

            return eval_

        legal_moves = list(board.legal_moves)

        if (maximizing):
            best_move = None
            best_val = float('-inf')
            for move in legal_moves:
                board.push(move)
                value = self.minimax_with_ab_pruning(False, board, depth + 1, alpha, beta)
                board.pop()

                if (value > best_val):
                    best_val = value
                    best_move = move

                alpha = max(alpha, best_val)

                if (beta <= alpha):
                    break

            if (depth > 0):
                return best_val
            else:
                return best_val, best_move
        else:
            best_move =  None
            best_val = float('inf')

            for move in legal_moves:
                board.push(move)
                value = self.minimax_with_ab_pruning(True, board, depth + 1, alpha, beta)
                board.pop()

                if (value < best_val):
                    best_val = value
                    best_move = move

                beta = min(beta, best_val)
                if (beta <= alpha):
                    break

            if (depth > 0):
                return best_val
            else:
                return best_val, best_move       

        

    def get_best_move(self, chess_board):
        fen_str = chess_board.fen()
        if (" w " in fen_str):
            maximizing = False
        else:
            maximizing = True

        #eval_, move = self.minimax(not maximizing, chess_board, 0)
        eval_, move = self.minimax_with_ab_pruning(not maximizing, chess_board, 0, float('-inf'), float('inf'))

        return eval_, move

