from typing import List, Tuple, TypeVar

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch as th

import os

regressive_bert_style = False

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
  max_validation_examples = 64,
  model_id: str = "experiment", tag: str = "model",
  store_path = None
):
  if not store_path:
    store_path = f"completed-experiments/{model_id}/{tag}.pt"
  (validation_batch_input, validation_batch_target, validation_batch_mask) = (
    validation_x[:max_validation_examples,:],
    validation_y[:max_validation_examples],
    validation_mask[:max_validation_examples] if validation_mask is not None else None
  )

  loss_fn = model.loss_fn()
  model.train()
  losses = []
  accuracies = []
  validation_accuracies = []
  for epoch in range(epochs):
    indices = np.random.permutation(range(len(train_x)))
    t = tqdm(range(0,(len(train_x)//batch_size)+1))
    for i in t:
      batch_indices = indices[i*batch_size:(i+1)*batch_size]
      if len(batch_indices) == 0:
        continue

      (batch_input, batch_target, batch_mask) = (
        train_x[batch_indices],
        train_y[batch_indices],
        train_mask[batch_indices] if train_mask is not None else None
      )

      optimizer.zero_grad(set_to_none=True)
      loss, accuracy = model.run_batch(
        device, loss_fn,
        batch_input, batch_target, batch_mask
      )
      losses.append(loss.item())
      accuracies.append(accuracy)
      loss.backward()
      optimizer.step()

      if i % 100 == 0:
        with th.no_grad():
          validation_loss, valid_accuracy = model.run_batch(
            device, loss_fn,
            validation_batch_input, validation_batch_target, validation_batch_mask
          )
          validation_loss = validation_loss.item()
          validation_accuracies.append(valid_accuracy)
      t.set_description(f"Epoch: {epoch} Iteration: {i} Loss: {np.mean(losses[-20:]):.3f} Validation Loss: {validation_loss:.3f} Accuracy: {np.mean(accuracies[-10:]):.3f} Validation Accuracy: {np.mean(validation_accuracies[-10:]):.3f}")
    
    # save your latest model
    th.save(model.state_dict(), store_path)

    #     bert_output_dir = f"./bert_model_save_from_training/{model_id}/"
    #     # Create output directory if needed
    #     if not os.path.exists(bert_output_dir):
    #       os.makedirs(bert_output_dir)

    #     if hasattr(model, "transformer"):
    #       print(f"Saving BERT model to {bert_output_dir}")
    #       bert = model.transformer
    #       model_to_save = bert.module if hasattr(bert, 'module') else bert  # Take care of distributed/parallel training
    #       model_to_save.save_pretrained(bert_output_dir)
    #       bert.config.save_pretrained(bert_output_dir)
  model.eval()
  return accuracies, validation_accuracies
