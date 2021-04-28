from typing import List, Tuple, TypeVar

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch as th

import os

T = TypeVar("T")
def split_train_validation(source_data: List[T], val_percent: float) -> Tuple[List[T], List[T], List[T], List[T]]:
  X = [i[0] for i in source_data]
  y = [i[1] for i in source_data]

  return train_test_split(X, y, test_size=val_percent)

def pad_text(text: List[int], to_length: int, with_index: int) -> List[int]:
  return text + [with_index] * (to_length - len(text))

batch_to_torch = lambda b_in,b_target,b_mask: (th.tensor(b_in, dtype=th.long),
                                        th.tensor(b_target, dtype=th.float if False else th.long),
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
  if False:
    (validation_batch_input, validation_batch_target, validation_batch_mask) = batch_to_torch(validation_x[:64,:], validation_y[:64], validation_mask[:64])
  else:
    (validation_batch_input, validation_batch_target, validation_batch_mask) = batch_to_torch(validation_x[:64,:], (validation_y[:64] - 1) * 2, validation_mask[:64])
  (validation_batch_input, validation_batch_target, validation_batch_mask) = list_to_device((validation_batch_input, validation_batch_target, validation_batch_mask), device)

  if False:
    loss_fn = th.nn.MSELoss()
  else:
    loss_fn = th.nn.CrossEntropyLoss()
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
          if False:
            (batch_input, batch_target, batch_mask) = batch_to_torch(batch_input, batch_target, batch_mask)
          else:
            (batch_input, batch_target, batch_mask) = batch_to_torch(batch_input, (batch_target - 1) * 2, batch_mask)
          (batch_input, batch_target, batch_mask) = list_to_device((batch_input, batch_target, batch_mask), device)
          
          if False:
            prediction = th.flatten(model(batch_input, batch_mask))
            loss = loss_fn(prediction, batch_target)
          else:
            prediction = model(batch_input, batch_mask)
            loss = loss_fn(prediction, batch_target)
          losses.append(loss.item())
          if False:
            accuracy = th.mean(th.eq(th.round(prediction * 2), th.round(batch_target * 2)).float())
          else:
            max_prediction = th.argmax(prediction, dim=1)
            accuracy = th.mean(th.eq(max_prediction, batch_target).float())
          accuracies.append(accuracy.item())

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          if i % 100 == 0:
            model.eval()
            if False:
              validation_prediction = th.flatten(model(validation_batch_input, validation_batch_mask))
              validation_loss = loss_fn(validation_prediction, validation_batch_target).item()
            else:
              validation_prediction = model(validation_batch_input, validation_batch_mask)
              validation_loss = loss_fn(validation_prediction, validation_batch_target).item()
            # print(validation_prediction)
            if False:
              validation_accuracy = th.mean(th.eq(th.round(validation_prediction * 2), th.round(validation_batch_target * 2)).float())
            else:
              max_validation_prediction = th.argmax(validation_prediction, dim=1)
              validation_accuracy = th.mean(th.eq(max_validation_prediction, validation_batch_target).float())
            validation_accuracies.append(validation_accuracy.item())
            model.train()
          t.set_description(f"Epoch: {epoch} Iteration: {i} Loss: {np.mean(losses[-20:]):.3f} Validation Loss: {validation_loss:.3f} Accuracy: {np.mean(accuracies[-10:]):.3f} Validation Accuracy: {np.mean(validation_accuracies[-10:]):.3f}")
      # save your latest model
      th.save(model.state_dict(), f"model_{model_id}.pt")

      bert_output_dir = './bert_model_save_from_training/'
      # Create output directory if needed
      if not os.path.exists(bert_output_dir):
          os.makedirs(bert_output_dir)

      print("Saving BERT model to %s" % bert_output_dir)
      bert = model.transformer
      model_to_save = bert.module if hasattr(bert, 'module') else bert  # Take care of distributed/parallel training
      model_to_save.save_pretrained(bert_output_dir)
      bert.config.save_pretrained(bert_output_dir)
  return accuracies, validation_accuracies