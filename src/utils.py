
import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split


class Preprocessing:
	
	def __init__(self):
		self.max_len = 30
		self.max_words = 500
		self.data = 'data/train.csv'
		self.test_size = 0.15
		
	def load_data(self):
		df = pd.read_csv(self.data)
		df.drop(['id','keyword','location'], axis=1, inplace=True)
		
		X = df['text'].values
		Y = df['target'].values
		
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=self.test_size)
		
		pass
		
	def prepare_tokens(self):
		self.tokens = Tokenizer(num_words=self.max_words)
		self.tokens.fit_on_texts(self.x_train)
		
		pass

	def sequence_to_token(self, x):
		sequences = self.tokens.texts_to_sequences(x)
		return sequence.pad_sequences(sequences, maxlen=self.max_len)