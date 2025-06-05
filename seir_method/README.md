### Declaration
The models provided in this code repo are for demonstration purposes.
Language model, vision-language model, and generative model can be replaced with whatever you wish.

### Config file
`utils/config.ini` : Model path and other parameters, you need to modify them before using.

### SEIR method
Start command:
```shell
cd seir_method
CUDA_VISIBLE_DEVICES=2,3 python seir_method.py \
--json_file_basename "data_test" \
--output_parent_dir "./t2ti_data" \
--dataset_id "00000" \
--conversation_turn 5 \
--mod_q_suggestion_num 3 \
--mod_ac_suggestion_num 3 \
--mod_c_suggestion_num 3 \
--dataset_size 10000 \
--save_batch_size 10 
```

Parameter Explanation:
```
json_file_basename: Specify the name of the JSON file to process.
output_parent_dir: jsonl and images output parent dir.
dataset_id: output child dir. The final output directory for current dataset is `output_parent_dir/dataset_id`.
conversation_turn: For per dataset sample, randomly select a number from 1 to conversation_turn as the number of turns in a multi-turn conversation.
mod_q_suggestion_num: Number of modifications for question.
mod_ac_suggestion_num: Number of modifications for answer.
mod_c_suggestion_num: Number of modifications for caption.
dataset_size: Number of dataset samples to generate.
save_batch_size: Number of batch size to save.
```