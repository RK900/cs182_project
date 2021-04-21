from typing import List, Tuple, TypeVar

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch as th

T = TypeVar("T")
def split_train_validation(source_data: List[T], val_percent: float) -> Tuple[List[T], List[T], List[T], List[T]]:
  X = [i[0] for i in source_data]
  y = [i[1] for i in source_data]

  return train_test_split(X, y, test_size=val_percent)

def pad_text(text: List[int], to_length: int, with_index: int) -> List[int]:
  return text + [with_index] * (to_length - len(text))

batch_to_torch = lambda b_in,b_target,b_mask: (th.tensor(b_in, dtype=th.long),
                                        th.tensor(b_target, dtype=th.float),
                                        th.tensor(b_mask, dtype=th.float))

def list_to_device(all_tensors, device):
  return [tensor.to(device) for tensor in all_tensors]

Example = Tuple[str, float]
def run_training_loop(
  model, optimizer, device,
  batch_size: int, epochs: int,
  train_x: List[str], train_mask, train_y: List[float],
  validation_x: List[str], validation_mask, validation_y: List[float],
  model_id: str = "experiment"
):
  (validation_batch_input, validation_batch_target, validation_batch_mask) = batch_to_torch(validation_x[:64,:], validation_y[:64], validation_mask[:64])
  (validation_batch_input, validation_batch_target, validation_batch_mask) = list_to_device((validation_batch_input, validation_batch_target, validation_batch_mask), device)

  loss_fn = th.nn.MSELoss()
  model.train()
  losses = []
  accuracies = []
  validation_accuracies = []
  for epoch in range(epochs):
      indices = np.random.permutation(range(len(train_x)))
      t = tqdm(range(0,(len(train_x)//batch_size)+1))
      for i in t:
          batch_indices = indices[i*batch_size:(i+1)*batch_size]
          (batch_input, batch_target, batch_mask) = (
            train_x[batch_indices],
            train_y[batch_indices],
            train_mask[batch_indices]
          )
          (batch_input, batch_target, batch_mask) = batch_to_torch(batch_input, batch_target, batch_mask)
          (batch_input, batch_target, batch_mask) = list_to_device((batch_input, batch_target, batch_mask), device)
          
          prediction = th.flatten(model(batch_input, batch_mask))
          loss = loss_fn(prediction, batch_target)
          losses.append(loss.item())
          accuracy = th.mean(th.eq(th.round(prediction * 2), th.round(batch_target * 2)).float())
          accuracies.append(accuracy.item())

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          if i % 10 == 0:
              model.eval()
              validation_prediction = th.flatten(model(validation_batch_input, validation_batch_mask))
              validation_loss = loss_fn(validation_prediction, validation_batch_target).item()
              # print(validation_prediction)
              validation_accuracy = th.mean(th.eq(th.round(validation_prediction * 2), th.round(validation_batch_target * 2)).float())
              validation_accuracies.append(validation_accuracy.item())
              model.train()
              t.set_description(f"Epoch: {epoch} Iteration: {i} Loss: {np.mean(losses[-20:]):.3f} Validation Loss: {validation_loss:.3f} Accuracy: {np.mean(accuracies[-10:]):.3f} Validation Accuracy: {np.mean(validation_accuracies[-10:]):.3f}")
      # save your latest model
      th.save(model.state_dict(), f"model_{model_id}.pt")
  return accuracies, validation_accuracies