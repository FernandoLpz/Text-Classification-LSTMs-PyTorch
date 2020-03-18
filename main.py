from src import PrepareData
from src import TextClassifier

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score

class ExecuteModel:
   def __init__(self, data):
      
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
      
      self.x_train = data.x_train
      self.x_test = data.x_test
      self.y_train = data.y_train
      self.y_test = data.y_test
      
      self.embeddings = data.embeddings
      
      self.seq_len = 140
      self.num_layers = 5
      self.hidden_dim = 64
      self.batch_size = 1
      self.embedding_size = 300
      
   def initialize_embeddings(self):
      
      self.embeddings = torch.from_numpy(self.embeddings).type(torch.FloatTensor)
      embedding = nn.Embedding.from_pretrained(self.embeddings)
      
      return embedding

      
   def init_train(self):
      
      predictions = list()
   
      embedding = self.initialize_embeddings()
      
      self.model = TextClassifier(self.seq_len, self.embedding_size, self.hidden_dim, self.batch_size, self.num_layers).to(self.device)

      optimizer = optim.RMSprop(self.model.parameters(), lr=0.001)
      
      for epoch in range(100):
         
         self.model.train()
         
         for sequence, target in zip(self.x_train, self.y_train):

            hc = self.model.init_hidden()
            
            x = torch.from_numpy(sequence).type(torch.LongTensor)
            y = torch.from_numpy(target).type(torch.LongTensor)
            
            x = embedding(x)
            
            x = x.reshape(x.shape[0], 1, x.shape[1])

            y_pred = self.model(x, hc)

            loss = F.binary_cross_entropy(y_pred, y.float())
            
            loss.backward()

            optimizer.step()
            
            optimizer.zero_grad()
            
            predictions.append(y_pred.detach().numpy)
        
         roc_auc = roc_auc_score(self.y_train, predictions)
         accuracy = accuracy_score(self.y_train, predictions)
         precision = precision_score(self.y_train, predictions)
         
         print("Epoch: %d, Loss %s , AUC: %.5f, ACC: %.5f, precision: %.5f" % (epoch+1, loss.item(), roc_auc, accuracy, precision))
            
         
            

   def prediction(self):
      
      self.model.eval()
      
      val_hc = self.net.init_hidden()
      auc_scores = list()
      with torch.no_grad():
         
         for i in range(int(len(self.x_test)/self.batch_size)):
            xval = self.char_embedding_siz(self.x_test[i*self.batch_size:(i+1)*self.batch_size])
            xval = np.reshape(xval, (xval.shape[1], self.batch_size, xval.shape[2]))
            
            xval = torch.from_numpy(xval).type(torch.FloatTensor)
            yval = torch.from_numpy(self.y_test[i*self.batch_size:(i+1)*self.batch_size]).type(torch.FloatTensor)
            
            y_pred = self.model(xval, val_hc)
            
            auc_scores.append(roc_auc_score(yval, y_pred))
            
         auc = sum(auc_scores) / len(auc_scores)
      
      return auc
   
if __name__ == "__main__":
   
   data = PrepareData()
   
   exeModel = ExecuteModel(data)   
   exeModel.init_train()