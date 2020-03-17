import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split

plt.style.use('seaborn-white')

class PrepareData:
   def __init__(self):
      self.build_dictionary()
      self.data_analitics()
      self.train_test_split_()
   
      pass
   
   def build_dictionary(self):
      
      self.dictionary = dict()
      
      with open('data/char_embeddigns.txt','r') as f:
         file = f.readlines()
      
      for item in file:
         item = item.strip().split()
         self.dictionary[item[0]] = item[1:]
         
      embeddings = list()
      for emb in self.dictionary.values():
         embeddings.append(emb)
         
      embeddings = np.array(embeddings)
         
      pass
   
   def data_analitics(self):
      train_data = pd.read_csv('data/train.csv', delimiter=',')
      self.train_text = train_data['text'].tolist()
      self.target = train_data['target'].values
      
      test_data = pd.read_csv('data/test.csv', delimiter=',')
      self.test_text = test_data['text'].tolist()
      
      # Prints hist about tweets length
      train_lengths = [len(i) for i in self.train_text]
      test_lengths = [len(i) for i in self.test_text]
      
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
      
      self.train_text = [PrepareData.remove_spaces(sentence, self.dictionary) for sentence in self.train_text]
      self.test_text = [PrepareData.remove_spaces(sentence, self.dictionary) for sentence in self.test_text]
      
      lengths_train = list()
      lengths_test = list()
      
      for sentence in self.train_text:
         lengths_train.append(len(sentence))
      
      for sentence in self.test_text:
         lengths_test.append(len(sentence))
      
      pass
      
   
   def train_test_split_(self):
      self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.train_text, self.target, test_size=0.20, random_state=42)
      
      self.y_train = np.array(self.y_train)
      self.y_test = np.array(self.y_test)
      
      self.y_train = np.reshape(self.y_train, (self.y_train.shape[0], 1))
      self.y_test = np.reshape(self.y_test, (self.y_test.shape[0], 1))
      
      pass
   
   @staticmethod
   def remove_spaces(sentence, dictionary):
      
      simbols = ['!','@','#','/','\\','-','_',':',',',"'",'.', '&', ';','[', ']','|', '(', ')', '%', '=', '*', '~', '$', '}', '+', '{', '^', '`', '>']
      sentence = [s.lower() for s in sentence if s in dictionary.keys() and s not in simbols]
      
      return sentence 
   