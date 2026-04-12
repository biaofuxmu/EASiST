
export WANDB_DISABLED=True

template=llama3
base_model_path=meta-llama/Llama-3.1-8B-Instruct

dataset1="easist_simulmt_data.en-de"
dataset2="easist_simulmt_data.en-es"
dataset3="easist_mt_data.en-de"
dataset4="easist_mt_data.en-es"

dataset="${dataset1},${dataset2},${dataset3},${dataset4}"

save_path=/path/to/save_models/easist_stage1_simulmt_model

mkdir -p "${save_path}"

{
  echo "${base_model_path}"
  echo "${save_path}"
  echo "${dataset}"

  llamafactory-cli train \
      --deepspeed examples/deepspeed/ds_z3_config.json \
      --ddp_timeout 180000000 \
      --flash_attn fa2 \
      --resize_vocab \
      --model_name_or_path "${base_model_path}" \
      --stage sft \
      --do_train \
      --finetuning_type full \
      --dataset "${dataset}" \
      --template "${template}" \
      --cutoff_len 1024 \
      --overwrite_cache \
      --preprocessing_num_workers 32 \
      --output_dir "${save_path}" \
      --logging_steps 1 \
      --save_steps 0.1 \
      --plot_loss true \
      --overwrite_output_dir \
      --per_device_train_batch_size 16 \
      --gradient_accumulation_steps 2 \
      --learning_rate 1e-5 \
      --num_train_epochs 1 \
      --lr_scheduler_type cosine \
      --warmup_ratio 0.1 \
      --bf16 True
} 2>&1 | tee "${save_path}/train.log"
