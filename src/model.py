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
      
      self.lstm_1 = nn.LSTMCell(input_size=self.vocab_size, hidden_size=self.hidden_dim)
      self.lstm_2 = nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
      self.dropout = nn.Dropout(p=0.5)
      
      self.fc = nn.Linear(in_features=self.hidden_dim, out_features=1)
      
   def forward(self, x, hc):
      
      output_seq = torch.empty((self.sequence_len, self.batch_size, self.vocab_size))

      hc_1, hc_2 = hc, hc
      
      # for every time step in the sequence
      for t in range(self.sequence_len):
         
         hc_1 = self.lstm_1(x[t], hc_1)
         h_1, c_1 = hc_1
         
         hc_2 = self.lstm_2(h_1, hc_2)
         h_2, c_2 = hc_2
   
      output = self.fc(self.dropout(h_2))
      output = F.relu(output)

      return output
   
   def init_hidden(self):
      return (torch.zeros(self.batch_size, self.hidden_dim), torch.zeros(self.batch_size, self.hidden_dim))