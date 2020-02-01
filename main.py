from src import PrepareData
from src import SiameseNet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import accuracy_score

class ExecuteModel:
   def __init__(self, data):
      
      self.x_train = data.x_train
      self.x_test = data.x_test
      self.y_train = data.y_train
      self.y_test = data.y_test
      
      # self.y_train = np.reshape(self.y_train, (self.y_train.shape[0],1))
      # self.y_test = np.reshape(self.y_test, (self.y_test.shape[0],1))
      
      # self.seq_len = 160
      # self.hidden_dim = 128
      # self.char_embedding_size = 300
      # self.batch_size = 128
   
   
      
   # def init_train(self):
   
   #    net = SiameseNet(sequence_len=self.seq_len, vocab_size=self.char_embedding_size, hidden_dim=self.hidden_dim, batch_size=self.batch_size)

   #    optimizer = optim.Adam(net.parameters(), lr=0.001)
      
   #    for epoch in range(10):
   #       net.train()
         
   #       hc = net.init_hidden()
         
   #       for i in range(int(len(self.x_train)/self.batch_size)):
            
   #          x = PrepareData.char_to_embedding(self.pDat.dictionary, self.x_train[i*self.batch_size:(i+1)*self.batch_size], np.zeros((self.batch_size, self.seq_len, self.char_embedding_size)))
   #          x = np.reshape(x, (x.shape[1], self.batch_size, x.shape[2]))
            
   #          x = torch.from_numpy(x).type(torch.FloatTensor)
   #          y = torch.from_numpy(self.y_train[i*self.batch_size:(i+1)*self.batch_size]).type(torch.FloatTensor)
            
   #          optimizer.zero_grad()
            
   #          output = net(x, hc)

   #          loss = F.binary_cross_entropy(output, y)
            
   #          loss.backward()
         
   #          optimizer.step()
            
   #          if epoch % 2 == 0: 
               
   #             test_hc = net.init_hidden()
               
   #             for j in range(int(len(self.x_test)/self.batch_size)):
                     
   #                xt = PrepareData.char_to_embedding(pDat.dictionary, self.x_test[j*self.batch_size:(j+1)*self.batch_size])
   #                xt = np.reshape(xt, (xt.shape[1], self.batch_size, xt.shape[2]))
                  
   #                xt = torch.from_numpy(xt).type(torch.FloatTensor)
   #                yt = torch.from_numpy(self.y_test[j*self.batch_size:(j+1)*self.batch_size]).type(torch.FloatTensor)
                  
   #                test_output = net(xt, test_hc)
   #                test_loss = F.binary_cross_entropy(test_output, yt)
                  
   #             print("Epoch: {}, Train Loss: {:.6f}, Test Loss: {:.6f}".format(epoch, loss.item(), test_loss.item()))


if __name__ == "__main__":
   
   data = PrepareData()
   exeModel = ExecuteModel(data)   
   # exeModel.init_train()