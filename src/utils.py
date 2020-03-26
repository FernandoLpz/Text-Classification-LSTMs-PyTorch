import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split

plt.style.use('seaborn-white')

class PrepareData:
   def __init__(self):
      
      self.build_dictionary()
      self.char_to_id()
      self.data_analitics()
      self.train_test_split_()
   
      pass
   
   def build_dictionary(self):
      
      self.dictionary = dict()
      self.embeddings = list()
      
      with open('data/char_embeddigns.txt','r') as f:
         file = f.readlines()
      
      for item in file:
         item = item.strip().split()
         self.dictionary[item[0]] = item[1:]
         
      # Adding the padding term
      self.dictionary['pad'] = np.random.randn(300)
             
      for emb in self.dictionary.values():
         emb = [float(i) for i in emb]
         self.embeddings.append(emb)
         
      self.embeddings = np.array(self.embeddings)
         
      pass
   
   def char_to_id(self):
      
      self.ch_to_id = dict()
      
      for id,char in enumerate(self.dictionary.keys()):
         self.ch_to_id[char] = id
         
      pass
   
   def data_analitics(self):
      
      train_data = pd.read_csv('data/train.csv', delimiter=',')
      
      train_text = train_data['text'].tolist()
      self.target = train_data['target'].values
      
      test_data = pd.read_csv('data/test.csv', delimiter=',')
      test_text = test_data['text'].tolist()
      
      # Prints hist about tweets length
      train_lengths = [len(i) for i in train_text]
      test_lengths = [len(i) for i in test_text]
      
      train_mean = sum(train_lengths) / len(train_lengths)
      train_variance = sum([((length - train_mean) ** 2) for length in train_lengths]) / len(train_lengths)
      train_std = train_variance ** 0.5
      
      test_mean = sum(test_lengths) / len(test_lengths)
      test_variance = sum([((length - test_mean) ** 2) for length in test_lengths]) / len(test_lengths)
      test_std = test_variance ** 0.5
      
      print('train lengths (mean): ', train_mean)
      print('train length (std): ', train_std)
      
      print('\ntest lengths (mean): ', test_mean)
      print('test length (std): ', test_std)
      
      plt.hist(train_lengths, histtype='stepfilled', color='red', edgecolor='red', alpha=0.3)
      plt.hist(test_lengths, histtype='stepfilled', color='green', edgecolor='green', alpha=0.3)
      #plt.show()
      
      train_text = [PrepareData.remove_spaces(sentence, self.dictionary) for sentence in train_text]
      test_text = [PrepareData.remove_spaces(sentence, self.dictionary) for sentence in test_text]
      
      self.padded_train_text = list()
      self.padded_test_text = list()
      
      
      # Padding sentences
      maxim = 0
      for seq in train_text:
         if len(seq) > maxim:
            maxim = len(seq)
      
      # Padding sequences
      for seq in train_text:
         if len(seq) < maxim:
            while len(seq) < maxim:
               seq.append('pad')
            self.padded_train_text.append(seq)
         else:
            self.padded_train_text.append(seq)
         
      for seq in test_text:
         if len(seq) < maxim:
            while len(seq) < maxim:
               seq.append('pad')
         self.padded_test_text.append(seq)

      pass
      
   
   def train_test_split_(self):
      
      xtrain, xtest, ytrain, ytest = train_test_split(self.padded_train_text, self.target, test_size=0.20, random_state=42)
         
      self.y_train = np.reshape(ytrain, (ytrain.shape[0], 1))
      self.y_test = np.reshape(ytest, (ytest.shape[0], 1))
      
      self.x_train, self.x_test = list(), list()
      
      for sentence in xtrain:
         to_id = list()
         for char in sentence:
            to_id.append(self.ch_to_id[char])
         to_id = np.array(to_id)
         self.x_train.append(to_id)
         
      for sentence in xtest:
         to_id = list()
         for char in sentence:
            to_id.append(self.ch_to_id[char])
         to_id = np.array(to_id)
         self.x_test.append(to_id)
   
      self.x_train = np.array(self.x_train)
      self.x_test = np.array(self.x_test)
      
      pass
   
   @staticmethod
   def remove_spaces(sentence, dictionary):
      
      simbols = ['\\',"'", '&', ';','[', ']','|', '(', ')', '%', '=', '~', '}', '+', '{', '^', '`', '>']
      sentence = [s.lower() for s in sentence if s in dictionary.keys() and s not in simbols]
      
      return sentence 
   