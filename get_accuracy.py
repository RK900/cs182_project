import json
import sys

out_file = sys.argv[1]
ref_file = sys.argv[2]

correct_count = 0
total_count = 0

with open(ref_file, "r") as ref:
  id_to_rating = {}
  for line in ref:
    review = json.loads(line)
    id_to_rating[review["review_id"]] = review["stars"]
  with open(out_file, "r") as out:
    for line in out:
      review = json.loads(line)
      total_count += 1
      if review["predicted_stars"] == id_to_rating[review["review_id"]]:
        correct_count += 1

print(correct_count / total_count)