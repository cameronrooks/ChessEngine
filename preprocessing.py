import chess
import pandas
import numpy as np
import sys
import h5py
from sklearn.model_selection import train_test_split
from utils import get_eval, encode_board, encode_board_without_turnbit


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

		if (eval == 100 or eval == -100):
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
h5f_train = h5py.File('./data/TrainDataSparse2.h5', 'w')
h5f_test = h5py.File('./data/TestDataSparse2.h5', 'w')

h5f_train.create_dataset('boards', data = x_train)
h5f_train.create_dataset('labels', data = y_train)

h5f_test.create_dataset('boards', data = x_test)
h5f_test.create_dataset('labels', data = y_test)

h5f_train.close()
h5f_test.close()



print(board_list.shape)
print(label_list.shape)
