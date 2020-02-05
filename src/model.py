import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNet(nn.ModuleList):
   def __init__(self, **kwargs):
      super(SiameseNet, self).__init__()
      
      # init the meta parameters
      self.hidden_dim = kwargs['hidden_dim']
      self.batch_size = kwargs['batch_size']
      self.sequence_len = kwargs['sequence_len']
      self.vocab_size = kwargs['vocab_size']
      self.LSTM_layers = kwargs['LSTM_layers']
      
      self.lstm = nn.LSTM(input_size=self.vocab_size, 
                          hidden_size=self.hidden_dim, 
                          num_layers=self.LSTM_layers,
                          dropout=0.1)
      
      self.dropout = nn.Dropout(p=0.1)
      self.fc_1 = nn.Linear(in_features=self.hidden_dim, out_features=1)
      
   def forward(self, x, hc):

      # out: tensor of shape [seq_len, batch_size, hidden_dim]. This tensor contains all the outputs
      # for each LSTM cell
      # hidden: tuple which contains (all hidden states, all current states)
      out, hidden = self.lstm(x, hc)
      
      # We take the last output, we do not care the previous outputs
      out = out[-1, :, :]
      
      # Feed Forward Neural Net with relu as activation function
      # out = F.relu(self.fc(self.dropout(out)))
      
      out = F.relu(self.fc_1(out))
      
      return out
   
   def init_hidden(self):
      # Initialization of hidden staes
      # It is needed a tuple, element [0] is about hidden states
      # element [1] is about current states
      # In case of using nn.LSTM, the hidden and current state must be defined as: [LSTM_layers, batch_size, hidden_dim]
      # IN case of using nn.LSTMCell, hidden and current state must be defined as: [batch_size, hidden_dim]
      
      return (torch.randn(self.LSTM_layers, self.batch_size, self.hidden_dim), 
              torch.randn(self.LSTM_layers, self.batch_size, self.hidden_dim))