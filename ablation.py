import papermill as pm

matrix = [
  ("max_training_samples", [10000, 100000, 200000]),
  ("lstm_hidden_size", [512, 1024, 2048]),
  ("epochs", [20])
]

def get_id(params):
  study_str = ""
  for key, value in params.items():
    study_str += f"-{key}-{value}"
  return f"classification{study_str}"

def run_matrix(cur_matrix, cur_params):
  if len(cur_matrix) == 0:
    run_id = get_id(cur_params)
    out_path = f"completed-experiments/{run_id}.ipynb"
    cur_params["experiment_id"] = run_id
    print("\n" * 10)
    print(f"RUNNING STUDY: {run_id}")
    pm.execute_notebook("./model-training-sentence-lstm.ipynb", out_path, cur_params, log_output=True)
  else:
    key, options = cur_matrix[0]
    for option in options:
      new_params = cur_params.copy()
      new_params[key] = option
      run_matrix(cur_matrix[1:], new_params)

run_matrix(matrix, dict())
