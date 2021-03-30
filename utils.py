import pickle
import os.path

def memo_load(value, key: str):
  pkl_path = f"{key}.pkl"
  if os.path.exists(pkl_path):
    with open(pkl_path, "rb") as pkl_file:
      return pickle.load(pkl_file)
  else:
    got_value = value()
    with open(pkl_path, "wb") as pkl_file:
      pickle.dump(got_value, pkl_file)
    return got_value
