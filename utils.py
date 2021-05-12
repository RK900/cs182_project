import mmap
import hickle
import os.path
import os
import hashlib
from collections import defaultdict
import random

def memo_load(value, key: str):
  hkl_path = f"{key}.hkl"
  if os.path.exists(hkl_path):
    return hickle.load(hkl_path)
  else:
    computed = value()
    hickle.dump(computed, hkl_path, mode="w")
    return hickle.load(hkl_path)

def manual_memo(compute, store, load, folder):
  if not os.path.exists(folder):
    computed = compute()
    os.makedirs(folder)
    store(computed, folder)
    return computed
  else:
    return load(folder)

# def hash_key(key):
#   return str(hashlib.sha256(key.encode("utf-8")).hexdigest())[:8]

def hash_file(file):
  h  = hashlib.sha256()
  with open(file, 'rb') as f:
    with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as mm:
      h.update(mm)
  return str(h.hexdigest())[:8]

def split_equally(dataset, out_dataset_size):
  amount_to_sample = out_dataset_size // 5
  reviews_dict = defaultdict(list)

  for (text, review) in dataset:
    reviews_dict[review].append(text)

  out_dataset = []
  for i in [1.0, 2.0, 3.0, 4.0, 5.0]:
    if len(reviews_dict[i]) < amount_to_sample:
      sampled = reviews_dict[i]
    else:
      sampled = random.sample(reviews_dict[i], amount_to_sample)
    out_dataset.extend(list(map(lambda text: (text, i), sampled)))

  return out_dataset
