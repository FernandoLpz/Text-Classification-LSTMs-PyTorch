import torch
import torch.nn as nn
import torch.nn.functional as F

class TweetClassifier(nn.ModuleList):

	def __init__(self):
		super(TweetClassifier, self).__init__()
		self.batch_size = 128
		self.hidden_dim = 64
		self.LSTM_layers = 64
		self.vocab_size = 20
		
		self.dropout = nn.Dropout(0.5)
		self.lstm = nn.LSTM(input_size=self.vocab_size, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers)
		self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=258)
		self.fc2 = nn.Linear(in_features=258, out_features=1)
		
	def forward(self, x, hc):
	
		hidden, current = self.lstm(x, hc)
		hidden = self.dropout(hidden)
		out = torch.relu_(self.fc1(hidden))
		out = torch.sigmoid(self.fc2(out))
		out = out.view(self.batch_size, -1)
		
		return out
		
	def init_hidden(self):
		h = torch.zeros((self.LSTM_layers, self.batch_size, self.hidden_dim), dtype=torch.float, requires_grad=True)
		c = torch.zeros((self.LSTM_layers, self.batch_size, self.hidden_dim), dtype=torch.float, requires_grad=True)
		
		torch.nn.init.normal_(h, mean=0, std=0.001)
		torch.nn.init.normal_(c, mean=0, std=0.001)
		
		hidden_states = (h,c)
		
		return hidden_states