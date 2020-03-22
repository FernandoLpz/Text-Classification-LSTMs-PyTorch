from src import PrepareData
from src import TextClassifier

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import roc_auc_score

class ExecuteModel:
   def __init__(self, data):
      
      self.x_train = data.x_train
      self.x_test = data.x_test
      self.y_train = data.y_train
      self.y_test = data.y_test
      

      self.num_layers = 3
      self.hidden_dim = 128
      self.num_epochs = 30
      self.embedding_size = 300
      
      self.embeddings = self.initialize_embeddings(data.embeddings)
      
   def initialize_embeddings(self, embeddings):
      
      embeddings = torch.from_numpy(embeddings).type(torch.FloatTensor)
      embeddings = nn.Embedding.from_pretrained(embeddings)
      
      return embeddings

      
   def init_train(self):
      
      self.model = TextClassifier(self.embedding_size, self.hidden_dim, self.num_layers)

      optimizer = optim.RMSprop(self.model.parameters(), lr=0.001)
      
      for epoch in range(self.num_epochs):
         
         self.model.train()
         
         predictions = list()
         
         for sequence, target in zip(self.x_train, self.y_train):

            hc = self.model.init_hidden()
            
            x = torch.from_numpy(sequence).type(torch.LongTensor)
            y = torch.from_numpy(target).type(torch.LongTensor)
            
            x = self.embeddings(x)
            
            x = x.reshape(x.shape[0], 1, x.shape[1])

            y_pred = self.model(x, hc)

            loss = F.binary_cross_entropy(y_pred, y.float())
            
            loss.backward()

            optimizer.step()
            
            optimizer.zero_grad()
            
            predictions.append(y_pred.detach().numpy())
            
         predictions = np.array(predictions)
         
         roc_auc = roc_auc_score(self.y_train, predictions)
         
         print("Epoch: %d, Loss %s , AUC: %.5f" % (epoch+1, loss.item(), roc_auc))
           

   def prediction(self):
      
      predictions = list()
      
      with torch.no_grad():
         
         self.model.eval()
         
         for sequence, target in zip(self.x_test, self.y_test):
            
            hc = self.net.init_hidden()
            
            x = torch.from_numpy(sequence).type(torch.LongTensor)
            y = torch.from_numpy(target).type(torch.LongTensor)
            
            x = self.embeddings(x)
            
            x = x.reshape(x.shape[0], 1, x.shape[1])
            
            y_pred = self.model(x, hc)
            
            loss = F.binary_cross_entropy(y_pred, y.float())
            
            predictions.append(y_pred.detach().numpy())
            
         predictions = np.array(predictions)
         
         roc_auc = roc_auc_score(self.y_test, predictions)
      
      return roc_auc, loss.item()
   
if __name__ == "__main__":
   
   data = PrepareData()
   
   exeModel = ExecuteModel(data)   
   exeModel.init_train()