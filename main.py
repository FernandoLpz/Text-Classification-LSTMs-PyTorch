import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score

from src import Preprocessing
from src import TweetClassifier

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyMapDataset(Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		
	def __len__(self):
		return len(self.x)
		
	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]
		

class Execute:

   def __init__(self):
      self.batch_size = 64
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
   
      training_set = MyMapDataset(self.x_train, self.y_train)
      test_set = MyMapDataset(self.x_test, self.y_test)
      
      loader_training = DataLoader(training_set, batch_size=self.batch_size)
      loader_test = DataLoader(test_set)
      
      optimizer = optim.RMSprop(self.model.parameters(), lr=0.01)
      
      for epoch in range(10):
      	
      	accuracy = 0
      	
      	self.model.train()

      	hc = self.model.init_hidden()
      	lote = 0
      	
      	for x_batch, y_batch in loader_training:
      		
      		
      		x = x_batch.type(torch.FloatTensor)
      		y = y_batch.type(torch.FloatTensor)
      		
      		try:
	      	  x = x.reshape(x.shape[1], self.batch_size, 1)
	      	except:
	      	  break
	      	
	      	y_pred = self.model(x, hc)
	      	
	      	loss = F.binary_cross_entropy(y_pred, y)
	      	
	      	loss.backward()
      		
      		optimizer.step()
      		
      		optimizer.zero_grad()
      		accuracy += torch.eq(y_pred.round(), y).float().mean()
      		lote += 1

      	accuracy = accuracy / lote
      	print("Epoch: %d, Loss %.5f , ACC: %.5f" % (epoch+1, loss.item(), accuracy))

   
if __name__ == "__main__":
   
   execute = Execute()
   execute.train()