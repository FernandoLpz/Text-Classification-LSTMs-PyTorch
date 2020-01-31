import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

plt.style.use('seaborn-white')

class PrepareData:
   def __init__(self):
      self.dictionary = self.build_dictionary()
      self.train_text = list()
      self.target = list()
      self.test_text = list()
      
      pass
   @staticmethod
   def data_split(text, target):
      return train_test_split(text, target, test_size=0.20, random_state=42)
   
   @staticmethod
   def remove_spaces(sentence, dictionary):
      #restricted = [' ', '\n', '\x89', 'Û', 'ª','å','Ê','Ò','Ï','¢','Ó','÷','\x9d','¤']
      
      sentence = [s for s in sentence if s in dictionary.keys()]
      
      return sentence

   @staticmethod
   def char_to_embedding(dictionary, sentences, sentence_embedded):
      
      i = 0
      for sentence in sentences:
         j = 0
         for char in sentence:
            sentence_embedded[i][j] = dictionary[char]
            j+=1
         i+=1

      return sentence_embedded   
   
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
      
   def build_dictionary(self):
      
      dictionary = dict()
      
      with open('data/char_embeddigns.txt','r') as f:
         file = f.readlines()
      
      for item in file:
         item = item.strip().split()
         dictionary[item[0]] = item[1:]
         
      return dictionary