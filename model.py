import numpy as np
from segtok import tokenizer
import torch as th
from torch import nn
from torch.nn.modules import rnn

from transformers import DistilBertForSequenceClassification, PretrainedConfig

# Using a basic RNN/LSTM for Language modeling
class ReviewPredictionModel(nn.Module):
    def __init__(self, vocab_size, rnn_size_token_count, num_layers=1, dropout=0):
        super().__init__()
        # self.rnn_size = rnn_size_token_count
        # self.embedding = nn.Embedding(vocab_size, rnn_size_token_count)
        # self.lstm = nn.LSTM(input_size=rnn_size_token_count, hidden_size=rnn_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        # self.linear = nn.Linear(rnn_size_token_count * 2, 1)
        self.token_count = rnn_size_token_count
        if False:
          self.transformer = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        else:
          self.transformer = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=9)
        
    def forward(self,x,x_mask):
        x = x[:,:self.token_count]
        x_mask = x_mask[:,:self.token_count]
        
        # embedded_input = self.embedding(x)
        # through_lstm,  _ = self.lstm(embedded_input)
        # lstm_forward_last = through_lstm[:,-1,:self.rnn_size]
        # lstm_backward_last = through_lstm[:,0,self.rnn_size:]

        if False:
          transfomer_logit = self.transformer(input_ids=x, attention_mask=x_mask, return_dict=True)["logits"][:,0]
          return th.sigmoid(transfomer_logit) * 4 + 1 # soft clipping to the review range
        else:
          transfomer_logit = self.transformer(input_ids=x, attention_mask=x_mask, return_dict=True)["logits"][:,:9]
          return transfomer_logit
