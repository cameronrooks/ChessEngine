# ChessEngine
This project is a chess engine that is based on deep learning methods. 

The engine consists of two parts - a machine learning model trained on labelled chess positions, as well as a minimax algorithm  that will attempt to find the best move in a chess position using that trained model.
I chose to use PyTorch to train my model due to the convenience provided by the DataLoader class, as well as the increased speed it provides over other alternatives, such as tensorflow.

The dataset used to train the model consisted of around 14 million chess positions represented in FEN notation, along with the Stockfish evaluation of those positions at a depth of 22.
The dataset that I used can be found here: https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations

The chess positions were one-hot encoded such that each set of 64 bits correspond to each different chess piece for each player, and whether it appears on one of the 64 squares on the chess board. In addition to this, bits for castling rights and a bit representing whose turn it is was appended to the end.
This resulted in each chess position being represented by a 773 bit vector.

The data was passed through a neural network consisting of 3 fully connected hidden layers with 2048 nodes in each layer, and it was trained for about 150 epochs as of right now.

Originally, I had intended on using a CNN to train the model, but there were a few pieces of information in the chess board that were not convenient to represent in a 2d tensor - namely castling rights and a turn bit.

After training, the minimax algorithm with alpha-beta pruning was implemented to play the game of chess using the information from the trained model.
After playing against a few different levels of the Stockfish chess engine, I have concluded that my chess engine has the playing strength roughly equivalent to a 900 to 1000 rated chess player. 

There are still many positions that the model does not correctly label, which results in it playing subpar moves. I believe that this issue is due in part to the dataset I am using. 14 million is only a very tiny fraction of the total amount of chess positions possible, so more data might improve the performance of the model. In addition, many of the games in the database come from master players. This means that not a lot of the games in the database have a large material imbalance (i.e. one side having more pieces than the other). Because of this, I do not believe that the model truly understands the value of all of the pieces on the board. I will likely try to fix this issue by using a dataset that contains games from both master players and amateur players, such as the monthly datasets that come from lichess.com.
