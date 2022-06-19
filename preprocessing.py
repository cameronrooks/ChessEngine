import chess
import pandas
import numpy as np
import sys
import h5py
from sklearn.model_selection import train_test_split

def get_eval(eval):
	#print(eval)


	if "#" in eval:
		if("-" in eval):
			return -100
		return 100
	elif "-" in eval:
		eval = int(eval)
	elif "+" in eval:
		eval = eval.lstrip("+")
		eval = int(eval)
	else:
		eval = int(eval)

	eval = eval/100
	return eval

def encode_board(fen):
	encoded_board = np.zeros((12, 8, 8), dtype = 'int8')
	counter = 0
	flip = 0

	#variables for castling rights and whose turn it is
	wck = 0
	wcq = 0
	bck = 0
	bcq = 0

	for i in range(len(fen)):
		char = fen[i]

		channel = 0
		sign = 1

		if char == " ":

			i = i + 1

			if (fen[i] == 'b'):
				to_move = 0
			else:
				to_move = 1

			i = i + 2

			for x in range(4):
				if(fen[i] == " " or fen[i] == "-"):
					break
				if(fen[i] == "Q"):
					wcq = 1
				if(fen[i] == "K"):
					wck = 1
				if(fen[i] == "q"):
					bcq = 1
				if(fen[i] == "k"):
					bck = 1

				i = i + 1
			break

		if char == "/":
			continue

		if char.isdigit():
			char = int(char)
			counter += char
			continue

		rank = counter//8
		file_ = counter%8

		counter += 1

		match char:
			case "p":
				channel = 0
				sign = 1
			case "P":
				channel = 1
				sign = 1
			case "n":
				channel = 2
				sign = 1
			case "N":
				channel = 3
				sign = 1
			case "r":
				channel = 4
				sign = 1
			case "R":
				channel = 5
				sign = 1
			case "b":
				channel = 6
				sign = 1
			case "B":
				channel = 7
				sign = 1
			case "q":
				channel = 8
				sign = 1
			case "Q":
				channel = 9
				sign = 1
			case "k":
				channel = 10
				sign = 1
			case "K":
				channel = 11
				sign = 1


		encoded_board[channel][rank][file_] = sign 


	encoded_board = encoded_board.flatten()
	encoded_board = np.append(encoded_board, [to_move, wck, wcq, bck, bcq])


	return encoded_board


def get_data(data, board_list, label_list):
	counter = 0
	for row in data:
		to_move = ""

		counter += 1
		fen = row[0]
		eval = row[1]

		if ("w" in fen):
			to_move = "w"
		else:
			to_move = "b"

		eval = get_eval(eval)

		if (eval == "SKIP"):
			continue

		board_list.append(encode_board(fen))
		label_list.append(eval)
		if counter%500000 == 0:
			print(counter)
			sys.stdout.flush()

	return board_list, label_list

data = pandas.read_csv('./data/chessData2.csv')
data2 = pandas.read_csv('./data/random_evals.csv')
data = data.to_numpy()
data2 = data2.to_numpy()

board_list = []
label_list = []

board_list, label_list = get_data(data, board_list, label_list)
board_list, label_list = get_data(data2, board_list, label_list)



board_list = np.array(board_list, dtype='int8')
label_list = np.array(label_list, dtype = 'float32')


#split dataset into test and train

train_size = int(.8 * board_list.shape[0])
test_size = board_list.shape[0] - train_size

x_train, x_test, y_train, y_test = train_test_split(board_list,label_list, test_size = test_size)
#y_train, y_test = train_test_split(label_list, test_size = test_size)

#board_list = np.split(board_list, [train_size])
#label_list = np.split(label_list, [train_size])

#x_train = board_list[0]
#x_test = board_list[1]
#y_train = label_list[0]
#y_test = label_list[1]


#save clean data to h5py files
h5f_train = h5py.File('./data/TrainDataSparse.h5', 'w')
h5f_test = h5py.File('./data/TestDataSparse.h5', 'w')

h5f_train.create_dataset('boards', data = x_train)
h5f_train.create_dataset('labels', data = y_train)

h5f_test.create_dataset('boards', data = x_test)
h5f_test.create_dataset('labels', data = y_test)

h5f_train.close()
h5f_test.close()



print(board_list.shape)
print(label_list.shape)
