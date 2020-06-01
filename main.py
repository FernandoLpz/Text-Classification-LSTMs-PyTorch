import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score

from src import Preprocessing
from src import TweetClassifier

class Execute:
   def __init__(self):
      self.__init_data__()
      self.model = TweetClassifier()
      
   def __init_data__(self):
   
      self.preprocessing = Preprocessing()
      self.preprocessing.load_data()
      self.preprocessing.prepare_tokens()

      raw_x_train = self.preprocessing.x_train
      raw_x_test = self.preprocessing.x_test
      self.y_train = self.preprocessing.y_train
      self.y_test = self.preprocessing.y_test

      self.x_train = self.preprocessing.sequence_to_token(raw_x_train)
      self.x_test = self.preprocessing.sequence_to_token(raw_x_test)
      
   def train(self):
   
      optimizer = optim.RMSprop(self.model.parameters(), lr=0.01)
      
      for epoch in range(10):
      	
      	predictions = list()
      	y_real = list()
         
      	self.model.train()
      	
      	for i in range(int(len(self.x_train) / 128)):
      	
      		x_batch = self.x_train[i * 128 : (i+1) * 128]
      		y_batch = self.y_train[i * 128 : (i+1) * 128]
      		
      		hc = self.model.init_hidden()
      		
      		x = torch.from_numpy(x_batch).type(torch.FloatTensor)
      		y = torch.from_numpy(y_batch).type(torch.FloatTensor)
      		
      		# x = embeddings(x)
      		x = x.reshape(1, 128, x.shape[1])
      		# print(x.shape)
      		
      		y_pred = self.model(x, hc)
      		
      		loss = F.binary_cross_entropy(y_pred, y.float())
      		
      		loss.backward()
      		
      		optimizer.step()
      		
      		optimizer.zero_grad()
      		
      		y_real += list(y.squeeze().detach().numpy())
      		predictions += list(y_pred.squeeze().detach().numpy())
      		
      	if epoch % 2 == 0:
      		train_auc = roc_auc_score(y_real, predictions)
      		print("Epoch: %d, Train Loss %.5f , Train AUC: %.5f" % (epoch+1, loss.item(), train_auc))

   
if __name__ == "__main__":
   
   execute = Execute()
   execute.train()