import numpy as np
from segtok import tokenizer
import torch as th
from torch import nn
from torch.nn.modules import rnn

from transformers import DistilBertForSequenceClassification, PretrainedConfig

batch_to_torch = lambda b_in,b_target,b_mask,regressive: (th.tensor(b_in, dtype=th.long),
                                        th.tensor(b_target, dtype=th.float if regressive else th.long),
                                        th.tensor(b_mask, dtype=th.float))

def list_to_device(all_tensors, device):
  return [tensor.to(device) for tensor in all_tensors]

class ReviewPredictionModel(nn.Module):
    def __init__(self, vocab_size, token_count, num_layers=1, dropout=0, regressive_bert_style = False):
        super().__init__()
        # self.rnn_size = rnn_size_token_count
        # self.embedding = nn.Embedding(vocab_size, rnn_size_token_count)
        # self.lstm = nn.LSTM(input_size=rnn_size_token_count, hidden_size=rnn_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        # self.linear = nn.Linear(rnn_size_token_count * 2, 1)
        self.regressive_bert_style = regressive_bert_style
        self.token_count = token_count
        if regressive_bert_style:
          self.transformer = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1)
        else:
          self.transformer = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=9)
        
    def forward(self,x,x_mask):
        x = x[:,:self.token_count]
        x_mask = x_mask[:,:self.token_count]
        
        # embedded_input = self.embedding(x)
        # through_lstm,  _ = self.lstm(embedded_input)
        # lstm_forward_last = through_lstm[:,-1,:self.rnn_size]
        # lstm_backward_last = through_lstm[:,0,self.rnn_size:]

        if self.regressive_bert_style:
          transfomer_logit = self.transformer(input_ids=x, attention_mask=x_mask, return_dict=True)["logits"][:,0]
          return th.sigmoid(transfomer_logit) * 4 + 1 # soft clipping to the review range
        else:
          transfomer_logit = self.transformer(input_ids=x, attention_mask=x_mask, return_dict=True)["logits"][:,:9]
          return transfomer_logit
    
    def run_batch(self, device, loss_fn, batch_input, batch_target, batch_mask):
        if self.regressive_bert_style:
            (batch_input, batch_target, batch_mask) = batch_to_torch(batch_input, batch_target, batch_mask, self.regressive_bert_style)
        else:
            (batch_input, batch_target, batch_mask) = batch_to_torch(batch_input, (batch_target - 1) * 2, batch_mask, self.regressive_bert_style)
        (batch_input, batch_target, batch_mask) = list_to_device((batch_input, batch_target, batch_mask), device)
        
        if self.regressive_bert_style:
            prediction = self(batch_input, batch_mask)
            loss = loss_fn(prediction, batch_target)
        else:
            prediction = self(batch_input, batch_mask)
            loss = loss_fn(prediction, batch_target)

        if self.regressive_bert_style:
            accuracy = th.mean(th.eq(th.round(prediction * 2), th.round(batch_target * 2)).float())
        else:
            max_prediction = th.argmax(prediction, dim=1)
            accuracy = th.mean(th.eq(max_prediction, batch_target).float())
        return loss, accuracy.item()
    
    def loss_fn(self):
        if self.regressive_bert_style:
            return th.nn.MSELoss()
        else:
            return th.nn.CrossEntropyLoss()
        

class DiscriminatorModel(nn.Module):
    def __init__(self, transformer, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.transformer = transformer
        
    def forward(self,x,x_mask):
#         x = x[:,:self.token_count]
#         x_mask = x_mask[:,:self.token_count]
        
        # embedded_input = self.embedding(x)
        # through_lstm,  _ = self.lstm(embedded_input)
        # lstm_forward_last = through_lstm[:,-1,:self.rnn_size]
        # lstm_backward_last = through_lstm[:,0,self.rnn_size:]

        if True:
          transfomer_logit = self.transformer(input_ids=x, attention_mask=x_mask, return_dict=True)["logits"][:,:self.num_labels]
          return transfomer_logit
    
    def run_batch(self, device, loss_fn, batch_input, batch_target, batch_mask):
        (batch_input, batch_target, batch_mask) = batch_to_torch(batch_input, batch_target, batch_mask)
        (batch_input, batch_target, batch_mask) = list_to_device((batch_input, batch_target, batch_mask), device)
        
        prediction = self(batch_input, batch_mask)
        loss = loss_fn(prediction, batch_target)

        max_prediction = th.argmax(prediction, dim=1)
        accuracy = th.mean(th.eq(max_prediction, batch_target).float())
        return loss, accuracy.item()
    
    def loss_fn(self):
        return th.nn.CrossEntropyLoss()