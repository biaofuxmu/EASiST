

MODEL_PATH=save_models/easist_stage3_simul_st_model
LANG=en-de # en-es
num_beams=1

INPUT_FILE=./data/offline_st/must-c.${LANG}.tst-COMMON.json
RESULT_PATH=${MODEL_PATH}/prediction_offline_must-c_${LANG}

mkdir -p ${RESULT_PATH}

python3 easist/offline_eval.py \
    --input_file ${INPUT_FILE} \
    --output_path ${RESULT_PATH} \
    --speech_llama $MODEL_PATH \
    --speech_model_type wav2vec_s \
    --instruction_prefix "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nYou are a professional translator, your task is to translate the given speech from {src_lang} into {tgt_lang}.\n" \
    --instruction_suffix "<|end-of-read|><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" \
    --num_beams 1 \
    --max_new_tokens 512
