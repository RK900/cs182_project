from typing import List, Tuple

import json

def load_dataset(file: str) -> List[Tuple[str, float]]:
  out = []
  with open(file, "r") as file:
    for line in file.readlines():
      parsed_line = json.loads(line)
      out.append((parsed_line["text"], parsed_line["stars"]))
  return out
