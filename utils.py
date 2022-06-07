import numpy as np

def encode_board(fen):
	encoded_board = np.zeros((6, 8, 8))
	counter = 0

	#variables for castling rights and whose turn it is
	to_move = 0
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
				to_move = -1
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


		encoded_board[channel][rank][file_] = sign



	
	encoded_board = encoded_board.flatten()
	encoded_board = np.append(encoded_board, [to_move, wck, wcq, bck, bcq])


	return encoded_board