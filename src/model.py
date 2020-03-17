import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNet(nn.ModuleList):
   def __init__(self, seq_len, embedding_size, hidden_dim, batch_size, num_layers):
      super(SiameseNet, self).__init__()
      
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
      
      # init the meta parameters
      self.sequence_len = seq_len
      self.batch_size = batch_size
      self.hidden_dim = hidden_dim
      self.LSTM_layers = num_layers
      self.vocab_size = embedding_size
      
      self.lstm = nn.LSTM(input_size=self.vocab_size, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers, dropout=0.1)
      self.fc = nn.Linear(in_features=self.hidden_dim, out_features=1)
      self.dropout = nn.Dropout(p=0.1)
      
   def forward(self, x, hc):

      # out: tensor of shape [seq_len, batch_size, hidden_dim]. This tensor contains all the outputs
      # for each LSTM cell
      # hidden: tuple which contains (all hidden states, all current states)
      out, hidden = self.lstm(x, hc)
      
      # We take the last output, we do not care the previous outputs
      out = out[-1, :, :]
      
      # Feed Forward Neural Net with relu as activation function
      # out = F.relu(self.fc(self.dropout(out)))
      
      out = F.relu(self.fc(out))
      
      return out
   
   def init_hidden(self):
      # Initialization of hidden staes
      # It is needed a tuple, element [0] is about hidden states
      # element [1] is about current states
      # In case of using nn.LSTM, the hidden and current state must be defined as: [LSTM_layers, batch_size, hidden_dim]
      # IN case of using nn.LSTMCell, hidden and current state must be defined as: [batch_size, hidden_dim]
      
      h = torch.zeros((self.LSTM_layers, self.batch_size, self.hidden_dim), dtype=torch.double, requires_grad=True)
      c = torch.zeros((self.LSTM_layers, self.batch_size, self.hidden_dim), dtype=torch.double, requires_grad=True)
      
      torch.nn.init.normal_(h, mean=0, std=0.001)
      torch.nn.init.normal_(c, mean=0, std=0.001)
      
      hidden_states = (h,c)
      
      return hidden_states