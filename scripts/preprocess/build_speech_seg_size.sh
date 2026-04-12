#!/usr/bin/env bash
set -euo pipefail

SPEECH_MODEL_PATH=/path/to/wav2vec2-or-wav2vec-s # facebook/wav2vec2-base-960h
INPUT_JSON=/path/to/input_simul_st.json
OUTPUT_JSON=data/simul_st/easist_simulst_data.en-de.json

python scripts/preprocess/build_speech_seg_size.py \
  --speech_model_path "${SPEECH_MODEL_PATH}" \
  --simul_path "${INPUT_JSON}" \
  --output_path "${OUTPUT_JSON}" \
  --main_context 32 \
  --right_context 16 \
  --speech_segment_size 400
