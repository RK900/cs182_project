import papermill as pm
import nbclient
import os

matrix = [
  ("balanced_categories", [False]),
  ("max_training_samples", [200000]),
  ("enable_orig", ["orig", "preprocess", "big"]),
  ("enable_aug", [False]),
  ("enable_aug3", [True, False]),
  ("sentence_pairs", ["3", True, False]),
  ("regressive_style_finetuning", [False]),
  ("lstm_bidi", [True, False]),
  ("lstm_hidden_size", [64, 256, 1024, 2048]),
  ("epochs", [20])
]

def is_valid_params(params):
  return params["enable_orig"] or params["enable_aug"] or params["enable_aug3"]

def get_id(params):
  study_str = ""
  for key, value in params.items():
    study_str += f"-{key}-{value}"
  return f"classification-sentence-embeddings{study_str}"

def run_matrix(cur_matrix, cur_params):
  if len(cur_matrix) == 0:
    if is_valid_params(cur_params):
      run_id = get_id(cur_params)
      out_path = f"completed-experiments/{run_id}.ipynb"
      if not os.path.exists(f"completed-experiments/{run_id}/main-accuracies.hkl"):
        cur_params["experiment_id"] = run_id
        print("\n" * 10)
        print(f"RUNNING STUDY: {run_id}")
        while True:
          try:
            pm.execute_notebook("./model-training-sentence-lstm.ipynb", out_path, cur_params, log_output=True)
            break
          except nbclient.exceptions.DeadKernelError:
            print("----------------------")
            print("RETRYING")
            print("----------------------")
  else:
    key, options = cur_matrix[0]
    for option in options:
      new_params = cur_params.copy()
      new_params[key] = option
      run_matrix(cur_matrix[1:], new_params)

if __name__ == "__main__":
  run_matrix(matrix, dict())
