import numpy as np



def get_eval_from_model_output(num, max_eval=50, min_eval=-50):
	eval = ((1 + num) / 2) * (max_eval - min_eval) + min_eval

	return eval


def get_eval(eval, max_eval = 50, min_eval = -50):

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

	if (eval > max_eval):
		eval = max_eval
	if (eval < min_eval):
		eval = min_eval

	#normalize the evals to be in the range [0,1]
	eval = 2 * ((eval - min_eval)/(max_eval - min_eval)) - 1

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