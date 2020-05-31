import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop

from keras.callbacks import EarlyStopping

class Preprocessing:
	
	def __init__(self):
		self.max_len = 20
		self.max_words = 1000
		self.data = 'data/train.csv'
		self.test_size = 0.15
		
	def load_data(self):
		df = pd.read_csv(self.data)
		df.drop(['id','keyword','location'], axis=1, inplace=True)
		
		X = df['text'].values
		Y = df['target'].values
		
		self.x_train, self.y_train, self.x_test, self.y_test = train_test_split(X, Y, test_size=self.test_size)
		
		pass
		
	def prepare_tokens(self):
		self.tokens = Tokenizer(num_words=self.max_words)
		self.tokens.fit_on_texts(self.x_train)
		
		pass

	def sequence_to_token(self, x):
		sequences = self.tokens.texts_to_sequences(x)
		return sequence.pad_sequences(sequences, maxlen=self.max_len)
		
class TweetClassifier(nn.ModuleList):

	def __init__(self):
		super(TweetClassifier, self).__init__()
		self.batch_size = 128
		self.hidden_dim = 64
		self.LSTM_layers = 64
		self.vocab_size = 1000
		
		self.dropout = nn.Dropout(0.5)
		self.lstm = nn.LSTM(input_size=self.vocab_size, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers, bidirectional=False)
		self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=258)
		self.fc2 = nn.Linear(in_features=258, out_features=1)
		
	def forward(self, x, hc):
	
		hidden, current = self.lstm(x, hc)
		hidden = hidden.squeeze()[-1, :]
		hidden = self.dropout(hidden)
		out = torch.relu_()(self.fc1(hidden))
		out = torch.sigmoid(self.fc2(out))
		
		out = out.view(self.batch_size, -1)
		
		return out
		
	def init_hidden(self):
		h = torch.zeros((self.LSTM_layers, self.batch_size, self.hidden_dim), dtype=torch.float, requires_grad=True)
		c = torch.zeros((self.LSTM_layers, self.batch_size, self.hidden_dim), dtype=torch.float, requires_grad=True)
		
		torch.nn.init.normal_(h, mean=0, std=0.001)
		torch.nn.init.normal_(c, mean=0, std=0.001)
		
		hidden_states = (h,c)
		
		return hidden_states
         
		
if __name__ == '__main__':
	preprocessing = Preprocessing()
	preprocessing.load_data()
	preprocessing.prepare_tokens()

	raw_x_train = preprocessing.x_train
	raw_x_test = preprocessing.x_test
	y_train = preprocessing.y_train
	y_test = preprocessing.y_test
	
	x_train = preprocessing.sequence_to_token(raw_x_train)
	x_test = preprocessing.sequence_to_token(raw_x_test)
	
	model = TweetClassifier()
	optimizer = optim.RMSprop(model.parameters(), lr=0.01)
	
	for epoch in range(10):
		
		predictions = list()
		y_real = list()
      
		model.train()
		
		for i in range(int(len(x_train) / 128)):
		
			x_batch = x_train[i * 128 : (i+1) * 128]
			y_batch = y_train[i * 128 : (i+1) * 128]
			
			hc = model.init_hidden()
			
			x = torch.from_numpy(x_batch).type(torch.LongTensor)
			y = torch.from_numpy(y_batch).type(torch.LongTensor)
			
			x = embeddings(x)
			x = x.reshape(x.shape[1], 128, x.shape[2])
			
			y_pred = model(x, hc)
			
			loss = F.binary_cross_entropy(y_pred, y.float())
			
			loss.backward()
			
			optimizer.step()
			
			optimizer.zero_grad()
			
			y_real += list(y.squeeze().detach().numpy())
			predictions += list(y_pred.squeeze().detach().numpy())
			
			if epoch % 2 == 0:
				train_auc = roc_auc_score(y_real, predictions)
				print("Epoch: %d, Train Loss %.5f , Train AUC: %.5f" % (epoch+1, loss.item(), train_auc))
				
	

# def RNN():
#     inputs = Input(name='inputs', shape=[max_len])

		# max_words = vocab_len
		# 50 = hidden_size
		# max_len = sequence_len
		
#     layer = Embedding(max_words,50,input_length=max_len)(inputs)
#     layer = LSTM(64)(layer)
#     layer = Dense(257,name='FC1')(layer)
#     layer = Activation('relu')(layer)
#     layer = Dropout(0.5)(layer)
#     layer = Dense(1,name='out_layer')(layer)
#     layer = Activation('sigmoid')(layer)
#     model = Model(inputs=inputs,outputs=layer)
#     return model
    
# model = RNN()
# model.summary()
# model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

# model.fit(sequences_matrix,Y_train,batch_size=64,epochs=30,
#           validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
          
# test_sequences = tok.texts_to_sequences(X_test)
# test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

# accr = model.evaluate(test_sequences_matrix,Y_test)
# print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))