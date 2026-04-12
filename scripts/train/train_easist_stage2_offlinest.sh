
llm_model=/path/to/save_models/easist_stage1_simulmt_model
speech_model=/path/to/wav2vec-S-large-lv60k-960h-ft # biaofu-xmu/wav2vec-S-Large-ft-960h

manifest_files="must-c.en-de.train.json,must-c.en-es.train.json"
data_root=./data/offline_st
eval_manifest_files=$(echo "${manifest_files}" | sed 's/train/dev/g')

save_path=save_models/easist_stage2_offline_st_model

mkdir -p "${save_path}"
chmod -R 777 "${save_path}"

{
  echo "${llm_model}"
  echo "${speech_model}"
  echo "${manifest_files}"
  echo "${eval_manifest_files}"
  echo "${save_path}"

  torchrun --standalone --nnodes=1 --nproc_per_node=8 \
      easist/train_easist_st.py \
      --deepspeed easist/config/ds_z2_config.json \
      --data "${data_root}" \
      --output_dir "${save_path}" \
      --manifest_files "${manifest_files}" \
      --eval_manifest_files "${eval_manifest_files}" \
      --instruction_prefix "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nYou are a professional simultaneous interpreter, your task is to translate the following streaming speech from {src_lang} into {tgt_lang}." \
      --instruction_suffix "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" \
      --task_type "offline_st" \
      --remove_unused_columns False \
      --seed 42 \
      --do_train True \
      --bf16 True \
      --learning_rate 2e-4 \
      --weight_decay 0.0 \
      --max_grad_norm 1.0 \
      --lr_scheduler_type cosine \
      --warmup_ratio 0.03 \
      --per_device_train_batch_size 4 \
      --gradient_accumulation_steps 4 \
      --num_train_epochs 6 \
      --llama_model "${llm_model}" \
      --speech_model "${speech_model}" \
      --frozen_modules "llm" \
      --speech_model_type wav2vec_s \
      --adapter_type blockconv \
      --disable_tqdm False \
      --logging_steps 10 \
      --save_steps 0.1 \
      --save_total_limit 10 \
      --report_to "tensorboard" \
      --per_device_eval_batch_size 8 \
      --eval_strategy steps \
      --eval_steps 0.1
} 2>&1 | tee "${save_path}/train.log"
