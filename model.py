import numpy as np
from segtok import tokenizer
import torch as th
from torch import nn
from torch.nn.modules import rnn

from transformers import DistilBertForSequenceClassification

# Using a basic RNN/LSTM for Language modeling
class ReviewPredictionModel(nn.Module):
    def __init__(self, vocab_size, rnn_size, num_layers=1, dropout=0):
        super().__init__()
        # self.rnn_size = rnn_size
        # self.embedding = nn.Embedding(vocab_size, rnn_size)
        # self.lstm = nn.LSTM(input_size=rnn_size, hidden_size=rnn_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        # self.linear = nn.Linear(rnn_size * 2, 1)
        self.transfomer = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        
    def forward(self,x,x_mask):
        x = x[:,:64]
        x_mask = x_mask[:,:64]
        transfomer_logit = self.transfomer(input_ids=x, attention_mask=x_mask, return_dict=True)["logits"][:,0]

        # embedded_input = self.embedding(x)
        # through_lstm,  _ = self.lstm(embedded_input)
        # lstm_forward_last = through_lstm[:,-1,:self.rnn_size]
        # lstm_backward_last = through_lstm[:,0,self.rnn_size:]
        return th.sigmoid(transfomer_logit) * 4 + 1 # soft clipping to the review range
