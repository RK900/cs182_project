import hickle
import os.path
import os
import hashlib

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

def hash_key(key):
  return str(hashlib.sha256(key.encode("utf-8")).hexdigest())[:8]
