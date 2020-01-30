from src import PrepareData
from src import SiameseNet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class ExecuteModel:
   def __init__(self):
      self.seq_len = 160
      self.hidden_dim = 128
      self.char_embedding_size = 300
      self.batch_size = 12
      
   @staticmethod
   def data_split(text, target):
      return train_test_split(text, target, test_size=0.20, random_state=42)
   
   def char_to_embedding(self, dictionary, sentences):
      sentence_embedded = np.zeros((self.batch_size, self.seq_len, self.char_embedding_size))
      
      i = 0
      for sentence in sentences:
         j = 0
         for char in sentence:
            sentence_embedded[i][j] = dictionary[char]
            j+=1
         i+=1

      return sentence_embedded
      
   def init_train(self):
      
      prepData = PrepareData()
      prepData.data_analitics()
      
      x_train, x_test, y_train, y_test = ExecuteModel.data_split(prepData.train_text, prepData.target)
      
      y_train = np.reshape(y_train, (y_train.shape[0],1))
      y_test = np.reshape(y_test, (y_test.shape[0],1))
      
      # compile the network - sequence_len, vocab_size, hidden_dim, batch_size
      net = SiameseNet(sequence_len=self.seq_len, vocab_size=self.char_embedding_size, hidden_dim=self.hidden_dim, batch_size=self.batch_size)
      
      # define the loss and the optimizer
      optimizer = optim.Adam(net.parameters(), lr=0.001)
      criterion = nn.BCELoss()
      
      for epoch in range(10):
         net.train()
         
         # reinit the hidden and cell steates
         hc = net.init_hidden()
         
         for i in range(int(len(x_train)/self.batch_size)):
            
            x = self.char_to_embedding(prepData.dictionary, x_train[i*self.batch_size:(i+1)*self.batch_size])
            x = np.reshape(x, (x.shape[1], self.batch_size, x.shape[2]))
            
            x = torch.from_numpy(x).type(torch.FloatTensor)
            y = torch.from_numpy(y_train[i*self.batch_size:(i+1)*self.batch_size]).type(torch.FloatTensor)
            
            optimizer.zero_grad()
            
            output = net(x, hc)

            loss = F.binary_cross_entropy(output, y)
            loss.backward()
         
            optimizer.step()
            
         # feedback every 10 epochs
         if epoch % 1 == 0: 
            #with torch.no_grad():
            net.eval()
            acc = list()
            
            # initialize the validation hidden state and cell state
            test_hc = net.init_hidden()
            for j in range(int(len(x_test)/self.batch_size)):
                  
                  xt = self.char_to_embedding(prepData.dictionary, x_test[j*self.batch_size:(j+1)*self.batch_size])
                  xt = np.reshape(xt, (xt.shape[1], self.batch_size, xt.shape[2]))
                  
                  xt = torch.from_numpy(xt).type(torch.FloatTensor)
                  yt = torch.from_numpy(y_test[j*self.batch_size:(j+1)*self.batch_size]).type(torch.FloatTensor)
                  
                  test_output = net(xt, test_hc)
                  test_loss = F.binary_cross_entropy(test_output, yt)
                  
                  acc.append(accuracy_score(yt, test_output))
         
            print("Epoch: {}, Batch: {}, Train Loss: {:.6f}, Test Loss: {:.6f}, Acc Test: {:.4f}".format(epoch, i, loss.item(), test_loss.item(), sum(acc)/len(acc)))



if __name__ == "__main__":
   
   exeModel = ExecuteModel()
   
   exeModel.init_train()