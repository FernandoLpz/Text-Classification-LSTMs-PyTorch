from src import PrepareData
from src import SiameseNet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

class ExecuteModel:
   def __init__(self, data):
      
      self.x_train = data.x_train
      self.x_test = data.x_test
      self.y_train = data.y_train
      self.y_test = data.y_test
      
      self.dictionary = data.dictionary
      
      self.seq_len = 160
      self.hidden_dim = 64
      self.char_embedding_size = 32
      self.batch_size = 64
      self.LSTM_layers = 4
      
   def char_to_embedding(self, sentences):
      
      sentence_embedded = np.zeros((self.batch_size, self.seq_len, self.char_embedding_size))
      
      i = 0
      for sentence in sentences:
         j = 0
         for char in sentence:
            sentence_embedded[i][j] = self.dictionary[char]
            j+=1
         i+=1

      return sentence_embedded  
   
   def prediction(self):
      self.net.eval()
      
      val_hc = self.net.init_hidden()
      auc_scores = list()
      with torch.no_grad():
         
         for i in range(int(len(self.x_test)/self.batch_size)):
            xval = self.char_to_embedding(self.x_test[i*self.batch_size:(i+1)*self.batch_size])
            xval = np.reshape(xval, (xval.shape[1], self.batch_size, xval.shape[2]))
            
            xval = torch.from_numpy(xval).type(torch.FloatTensor)
            yval = torch.from_numpy(self.y_test[i*self.batch_size:(i+1)*self.batch_size]).type(torch.FloatTensor)
            
            y_pred = self.net(xval, val_hc)
            
            auc_scores.append(roc_auc_score(yval, y_pred))
            
         auc = sum(auc_scores) / len(auc_scores)
      
      return auc
      
   def init_train(self):
   
      self.net = SiameseNet(sequence_len=self.seq_len, 
                            vocab_size=self.char_embedding_size, 
                            hidden_dim=self.hidden_dim, 
                            batch_size=self.batch_size,
                            LSTM_layers=self.LSTM_layers)

      optimizer = optim.Adam(self.net.parameters(), lr=0.001)
      
      for epoch in range(10):
         self.net.train()
         
         hc = self.net.init_hidden()
         
         for i in range(int(len(self.x_train)/self.batch_size)):
            
            x = self.char_to_embedding(self.x_train[i*self.batch_size:(i+1)*self.batch_size])
            x = np.reshape(x, (x.shape[1], self.batch_size, x.shape[2]))
            
            x = torch.from_numpy(x).type(torch.FloatTensor)
            y = torch.from_numpy(self.y_train[i*self.batch_size:(i+1)*self.batch_size]).type(torch.FloatTensor)
            
            optimizer.zero_grad()
            
            output = self.net(x, hc)

            loss = F.binary_cross_entropy(output, y)
            
            loss.backward()
         
            optimizer.step()
            
         if epoch % 1 == 0: 
            
            test_hc = self.net.init_hidden()
            
            for j in range(int(len(self.x_test)/self.batch_size)):
               
               xt = self.char_to_embedding(self.x_test[j*self.batch_size:(j+1)*self.batch_size])
               xt = np.reshape(xt, (xt.shape[1], self.batch_size, xt.shape[2]))
               
               xt = torch.from_numpy(xt).type(torch.FloatTensor)
               yt = torch.from_numpy(self.y_test[j*self.batch_size:(j+1)*self.batch_size]).type(torch.FloatTensor)
               
               test_output = self.net(xt, test_hc)
               
               test_loss = F.binary_cross_entropy(test_output, yt)
            
            auc = self.prediction()
               
            print("Epoch: {}, Train Loss: {:.6f}, Test Loss: {:.6f}, AUC: {:.6f}".format(epoch, loss.item(), test_loss.item(), auc))


if __name__ == "__main__":
   
   data = PrepareData()
   
   exeModel = ExecuteModel(data)   
   exeModel.init_train()