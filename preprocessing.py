import chess
import pandas
import numpy as np
import sys
import h5py

def get_eval(eval):
	#print(eval)

	if "#" in eval:
		return "SKIP"
	elif "-" in eval:
		eval = int(eval)
	elif "+" in eval:
		eval = eval.lstrip("+")
		eval = int(eval)

	return eval

def encode_board(fen):
	encoded_board = np.zeros((6, 8, 8))
	counter = 0

	for char in fen:
		channel = 0
		sign = 1

		if char == " ":
			break

		if char == "/":
			continue

		if char.isdigit():
			char = int(char)
			counter += char
			continue

		rank = counter//8
		file = counter%8

		counter += 1

		match char:
			case "p":
				channel = 0
				sign = -1
			case "P":
				channel = 0
				sign = 1
			case "n":
				channel = 1
				sign = -1
			case "N":
				channel = 1
				sign = 1
			case "r":
				channel = 2
				sign = -1
			case "R":
				channel = 2
				sign = 1
			case "b":
				channel = 3
				sign = -1
			case "B":
				channel = 3
				sign = 1
			case "q":
				channel = 4
				sign = -1
			case "Q":
				channel = 4
				sign = 1
			case "k":
				channel = 5
				sign = -1
			case "K":
				channel = 5
				sign = 1


		encoded_board[channel][rank][file] = sign

	return encoded_board


data = pandas.read_csv('./data/chessData2.csv')
data = data.to_numpy()


board_list = []
label_list = []
counter = 0

for row in data:
	counter += 1
	fen = row[0]
	eval = row[1]

	eval = get_eval(eval)

	if (eval == "SKIP"):
		continue

	board_list.append(encode_board(fen))
	label_list.append(eval)
	if counter%500000 == 0:
		print(counter)
		sys.stdout.flush()


board_list = np.array(board_list, dtype='int8')
label_list = np.array(label_list, dtype = 'int32')

h5f = h5py.File('./data/ChessData.h5', 'w')
h5f.create_dataset('boards', data = board_list)
h5f.create_dataset('labels', data = label_list)

h5f.close()



print(board_list.shape)
print(label_list.shape)
