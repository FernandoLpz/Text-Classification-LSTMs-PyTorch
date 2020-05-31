import torch
import torch.nn as nn
import torch.nn.functional as F

class TextClassifier(nn.ModuleList):
   def __init__(self, embedding_size, hidden_dim, num_layers, batch_size):
      super(TextClassifier, self).__init__()
      
      self.batch_size = batch_size
      self.hidden_dim = hidden_dim
      self.LSTM_layers = num_layers
      self.vocab_size = embedding_size
      
      self.dropout = nn.Dropout(0.5)
      
      self.cnn = nn.Conv1d(in_channels=136, out_channels=70, kernel_size=3, stride=2)
      self.lstm = nn.LSTM(input_size=self.vocab_size, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers, bidirectional=False)
      #self.fc1 = nn.Linear(in_features=self.hidden_dim , out_features=self.hidden_dim )
      self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=1)
      
   def forward(self, x, hc):

      # cnn_out = self.cnn(x)
      
      # cnn_out = self.dropout(cnn_out)
      
      # cnn_out = cnn_out.reshape(cnn_out.shape[1], self.batch_size, cnn_out.shape[2])
      # out: tensor of shape [seq_len, batch_size, hidden_dim]. This tensor contains all the outputs
      # for each LSTM cell
      # hidden: tuple which contains (all hidden states, all current states)
      hidden, current = self.lstm(x, hc)
      
      # We take the last output, we do not care the previous outputs
      # hidden = hidden[-1, :, :]
      # hidden = hidden.contiguous().view(-1, self.hidden_dim)

      hidden = hidden.squeeze()[-1, :]

      # Dropout
      hidden = self.dropout(hidden)
      
      # Feed Forward Neural Net with sigmoid as activation function
      # out = torch.tanh(self.fc1(hidden))
      out = torch.sigmoid(self.fc1(hidden))

      out = out.view(self.batch_size, -1)
      
      # Just remove one dimention
      #out = out[:,-1]

      return out
   
   def init_hidden(self):
      # Initialization of hidden staes
      # It is needed a tuple, element [0] is about hidden states
      # element [1] is about current states
      # In case of using nn.LSTM, the hidden and current state must be defined as: [LSTM_layers, batch_size, hidden_dim]
      # IN case of using nn.LSTMCell, hidden and current state must be defined as: [batch_size, hidden_dim]
      
      h = torch.zeros((self.LSTM_layers, self.batch_size, self.hidden_dim), dtype=torch.float, requires_grad=True)
      c = torch.zeros((self.LSTM_layers, self.batch_size, self.hidden_dim), dtype=torch.float, requires_grad=True)
      
      torch.nn.init.normal_(h, mean=0, std=0.001)
      torch.nn.init.normal_(c, mean=0, std=0.001)
      
      hidden_states = (h,c)
      
      return hidden_states