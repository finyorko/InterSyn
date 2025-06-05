# SynJudge

## Sample question
The sample questions are listed in sample_question.txt

## Score prompt
The score prompt is shown in socre_prompt.txt

## Usage
1. You need to construct the data that needs to be scored into the format of model_answer.jsonl.
2. Use gpt-4o_judge.py, internvl_judge.py and qwenvl_judge.py to generate score jsonl file.
3. Use compute_mean_var.py to compute the mean and variance.