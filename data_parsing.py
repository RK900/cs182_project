from typing import List, Tuple

import json

def load_dataset(file: str) -> List[Tuple[str, float]]:
  out = []
  with open(file, "r") as file:
    for line in file.readlines():
      parsed_line = json.loads(line)
      out.append((parsed_line["text"], parsed_line["stars"]))
  return out

def load_gen_dataset(file: str) -> List[Tuple[str, float]]:
  out = []
  with open(file, "r") as file:
    loaded = json.loads(json.load(file))
    for text in loaded.keys():
      out.append((text, loaded[text]))
  return out

def load_preprocessed_dataset(file: str) -> List[Tuple[str, float]]:
  out = []
  with open(file, "r") as file:
    loaded = json.loads(json.load(file))
    for elem in loaded:
      out.append((elem["text"], elem["stars"]))
  return out
