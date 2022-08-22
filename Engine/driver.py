import ChessEngine
import chess

engine = ChessEngine.ChessEngine("../trained_models/model1/epochs/epoch153")


board = chess.Board()

board.push_san("e4")
board.push_san("e5")


print(engine.get_best_move(board))