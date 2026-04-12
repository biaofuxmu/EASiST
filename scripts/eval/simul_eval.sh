export WANDB_DISABLED=True
export TF_ENABLE_ONEDNN_OPTS=0

MODEL_PATH=save_models/easist_stage3_simul_st_model
LANG=en-de # en-es

INPUT_FILE=./data/offline_st/must-c.${LANG}.tst-COMMON.json

echo ${MODEL_PATH}
echo ${INPUT_FILE}

for latency_prob in 0.1 0.2 0.3 0.4 0.5 0.6
do
    RESULT_PATH=${MODEL_PATH}/simul_results/prediction_simul_must-c_${LANG}_prob_${latency_prob}

    mkdir -p ${RESULT_PATH}

    echo ${RESULT_PATH}
    echo ${latency_prob}

    max_token=$(awk "BEGIN {print $latency_prob * 500}")

    if [ "${max_token}" -gt 1024 ]; then
        max_token=1024
    fi
    echo ${max_token}

    python3 easist/simul_eval.py \
        --data_path ${INPUT_FILE} \
        --output_dir ${RESULT_PATH} \
        --speech_llama $MODEL_PATH \
        --instruction "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nYou are a professional simultaneous interpreter, your task is to translate the following streaming speech from {src_lang} into {tgt_lang}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" \
        --num_beams 1 \
        --max_new_tokens ${max_token} \
        --latency_prob ${latency_prob}
done
