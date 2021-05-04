import numpy as np
from segtok import tokenizer
import torch as th
from torch import nn
from torch.nn.modules import rnn

batch_to_torch = lambda b_in,b_target: (th.tensor(b_in, dtype=th.float),
                                        th.tensor(b_target, dtype=th.float if False else th.long))

def list_to_device(all_tensors, device):
  return [tensor.to(device) for tensor in all_tensors]

class ReviewPredictionModel(nn.Module):
    def __init__(self, embedded_size, rnn_size, num_layers=1, dropout=0):
        super().__init__()
        self.rnn_size = rnn_size
        self.lstm = nn.LSTM(input_size=embedded_size, hidden_size=rnn_size, num_layers=num_layers, batch_first=True)
        self.linear1 = nn.Linear(embedded_size * 10, embedded_size * 5)
        self.linear2 = nn.Linear(embedded_size * 5, embedded_size * 2)
        self.linear3 = nn.Linear(embedded_size * 2, embedded_size * 1)
        self.finalLinear = nn.Linear(embedded_size * 1, 9)
        self.linear = nn.Linear(rnn_size, 9)

    def forward(self, x):
        through_lstm,  _ = self.lstm(x)
        return self.linear(through_lstm[:,-1,:])
#         x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))
#         x = th.relu(self.linear1(x))
#         x = th.relu(self.linear2(x))
#         x = th.relu(self.linear3(x))
#         return self.finalLinear(x)
        # return self.finalLinear(x[:, 0, :])
    
    def run_batch(self, device, loss_fn, batch_input, batch_target, batch_mask):
        (batch_input, batch_target) = batch_to_torch(batch_input, (batch_target - 1) * 2)
        (batch_input, batch_target) = list_to_device((batch_input, batch_target), device)
        
        prediction = self(batch_input)
        loss = loss_fn(prediction, batch_target)

        max_prediction = th.argmax(prediction, dim=1)
        accuracy = th.mean(th.eq(max_prediction, batch_target).float())
        return loss, accuracy.item()
    
    def loss_fn(self):
        return th.nn.CrossEntropyLoss()
