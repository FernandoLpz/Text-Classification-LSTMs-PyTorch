import argparse

def parameter_parser():

	 parser = argparse.ArgumentParser(description = "Tweet Classification")

	 parser.add_argument("--epochs",
								dest = "epochs",
								type = int,
								default = 10,
						 help = "Number of gradient descent iterations. Default is 200.")

	 parser.add_argument("--learning_rate",
								dest = "learning_rate",
								type = float,
								default = 0.01,
						 help = "Gradient descent learning rate. Default is 0.01.")

	 parser.add_argument("--hidden_dim",
								dest = "hidden_dim",
								type = int,
								default = 128,
						 help = "Number of neurons by hidden layer. Default is 128.")
						 
	 parser.add_argument("--lstm_layers",
								dest = "lstm_layers",
								type = int,
								default = 2,
					 help = "Number of LSTM layers")
					 
	 parser.add_argument("--batch_size",
									dest = "batch_size",
									type = int,
									default = 64,
							 help = "Batch size")

	 parser.add_argument("--test_size",
								dest = "test_size",
								type = float,
								default = 0.20,
						 help = "Size of test dataset. Default is 10%.")
						 
	 parser.add_argument("--max_len",
								dest = "max_len",
								type = int,
								default = 20,
						 help = "Maximum sequence length per tweet")
						 
	 parser.add_argument("--max_words",
								dest = "max_words",
								type = float,
								default = 1000,
						 help = "Maximum number of words in the dictionary")					 
	 return parser.parse_args()
