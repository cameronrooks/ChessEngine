import ChessEngine
import chess
import chess.pgn

engine = ChessEngine.ChessEngine("../trained_models/model1/epochs/epoch150", 4)
#engine = ChessEngine.ChessEngine("../trained_models/model2/epochs/epoch104")

board = chess.Board()
game =  chess.pgn.Game()

node = None

def play_as_black():
    count = 0
    while (not board.is_checkmate()):
        engine_move = engine.get_best_move(board)[1]

        print(engine_move)

        board.push(engine_move)


        if (count == 0):
            node = game.add_variation(engine_move)
        else:
            node = node.add_variation(engine_move)

        if (board.is_checkmate()):
            break

        player_move = input()
        try:
            parsed_move = board.parse_san(player_move)
        except:
            print("illegal move")
            continue
        board.push_san(player_move)
        node = node.add_variation(parsed_move)

        print(board)
        print('\n')


        count += 1


def play_as_white():
    count = 0
    while (not board.is_checkmate()):
        player_move = input()
        try:
            parsed_move = board.parse_san(player_move)
        except:
            print("illegal move")
            continue
        board.push_san(player_move)

        if (count == 0):
            node = game.add_variation(parsed_move)
        else:
            node = node.add_variation(parsed_move)

        if (board.is_checkmate()):
            break

        engine_move = engine.get_best_move(board)[1]
        node = node.add_variation(engine_move)

        print(engine_move)

        board.push(engine_move)
        print(board)
        print('\n')

        count += 1

def play_against_self():
    count = 0
    while (not board.is_checkmate() and count < 50):
        move = engine.get_best_move(board)[1]

        if (count == 0):
            node = game.add_variation(move)
        else:
            node = node.add_variation(move)
        board.push(move)
        print(board)

        count += 1


play_as_black()

print(game)