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
      
      self.num_layers = 4
      self.hidden_dim = 64
      self.num_epochs = 200
      self.batch_size = 2
      self.embedding_size = 300
      
      self.embeddings = self.initialize_embeddings(data.embeddings)
      
   def initialize_embeddings(self, embeddings):
      
      embeddings = torch.from_numpy(embeddings).type(torch.FloatTensor)
      embeddings = nn.Embedding.from_pretrained(embeddings)
      
      return embeddings

      
   def init_train(self):
      
      self.model = TextClassifier(self.embedding_size, self.hidden_dim, self.num_layers, self.batch_size)

      optimizer = optim.SGD(self.model.parameters(), lr=0.01)
      # optimizer = optim.RMSprop(self.model.parameters(), lr=0.01)
      
      for epoch in range(self.num_epochs):
         
         self.model.train()
         
         predictions = list()
         y_real = list()
         
         for i in range(int(self.x_train.shape[0] / self.batch_size)):
         
            x_batch = self.x_train[i * self.batch_size : (i+1) * self.batch_size]
            y_batch = self.y_train[i * self.batch_size : (i+1) * self.batch_size]
            
            hc = self.model.init_hidden()
            
            # x.shape = [batch_size, seq_len]
            # y.shape = [batch_size, 1]
            x = torch.from_numpy(x_batch).type(torch.LongTensor)
            y = torch.from_numpy(y_batch).type(torch.LongTensor)

            # x.shape = [batch_size, seq_len, embedding_size]
            x = self.embeddings(x)

            # x = x.reshape(x.shape[1], self.batch_size, x.shape[2])

            y_pred = self.model(x, hc)

            loss = F.binary_cross_entropy(y_pred, y.float())
            
            loss.backward()

            optimizer.step()
            
            optimizer.zero_grad()
            
            y_real += list(y.squeeze().detach().numpy())
            predictions += list(y_pred.squeeze().detach().numpy())
            break
         break
            
         # Show metrics every two epochs 
         if epoch % 2 == 0:
         
            train_auc = roc_auc_score(y_real, predictions)
            
            print("Epoch: %d, Train Loss: %.5f, Train AUC: %.5f" % (epoch, loss.item(), train_auc))
               
            # test_auc, test_loss = self.evaluation()
            
            # print("Epoch: %d, Train Loss %.5f , Test Loss: %.5f, Train AUC: %.5f, Test AUC: %.5f" % (epoch+1, loss.item(), test_loss, train_auc, test_auc))
           

   def evaluation(self):
      
      predictions = list()
      
      with torch.no_grad():
         
         self.model.eval()
         
         for sequence, target in zip(self.x_test, self.y_test):
            
            hc = self.model.init_hidden()
            
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