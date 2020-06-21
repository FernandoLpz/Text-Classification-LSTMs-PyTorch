# Text Classification through a LSTM-based model

The aim of this repository is to show a baseline model for text classification by implementing a LSTM-based model coded in PyTorch. In order to provide a better understanding of the model, it will be used a dataset of Tweets provided by Kaggle. 

## 1. Data
As it was mentioned above, the dataset we are woking with is about Tweets regarding fake news. The head of the dataset looks like this:

|id| text | target |
| ------------- | ------------- | ------------- |
| 1  | Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all  |1  |
| 2  | SOOOO PUMPED FOR ABLAZE ???? @southridgelife  | 0  |
| 3  | INEC Office in Abia Set Ablaze - http://t.co/3ImaomknnA  | 1 |
| 4  | Building the perfect tracklist to life leave the streets ablaze  | 0  |

This dataset can be found in ``data/tweets.csv``. 

## 2. The model
As it was already commented, the aim of this repository is to provide a base line model for text classfication. In this sense, the model is based on a two-stacked LSTM layers followed by two linear layers. The dataset is preprocessed through a tokens-based technique, then tokens are associated to an embedding layer. The following image describes the pipeline of the model.