import torch as th
from torch import nn

batch_to_torch = lambda b_in,b_target: (th.tensor(b_in, dtype=th.float),
                                        th.tensor(b_target, dtype=th.float if False else th.long))

def list_to_device(all_tensors, device):
  return [tensor.to(device) for tensor in all_tensors]

class ReviewPredictionModel(nn.Module):
    def __init__(self, embedded_size, rnn_size, num_layers=1, dropout=0, bidi_lstm = False):
        super().__init__()
        self.rnn_size = rnn_size
        self.bidi_lstm = bidi_lstm
        self.lstm = nn.LSTM(input_size=embedded_size, hidden_size=rnn_size, num_layers=num_layers, batch_first=True, bidirectional=bidi_lstm)

        self.linear = nn.Linear(rnn_size * 2 if bidi_lstm else rnn_size, 9)

    def forward(self, x):
        through_lstm,  _ = self.lstm(x)
        return self.linear(through_lstm[:,-1,:])
    
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
