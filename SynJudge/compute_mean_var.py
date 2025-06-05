import json
import numpy as np

# Input file path
input_file = "model_answer.jsonl"

# read JSONL
def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append(item["labels"])
    return np.array(data)

# Load data and convert it to a NumPy array
scores = read_jsonl(input_file)

# compute mean and variance
score_avg = scores.mean(axis=0).tolist()
variance = scores.var(axis=0).tolist()
score_all = float(np.mean(score_avg))
variance_all = float(np.mean(variance))

# Construct output result
result = {
    "score": score_avg,
    "variance": variance,
    "score_all": score_all,
    "variance_all": variance_all
}

# Save the output result to a file
output_path = "score.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print("The result has been saved to: ", output_path)