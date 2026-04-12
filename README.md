# EASiST: Efficient and Adaptive Simultaneous Speech Translation

[![Paper](https://img.shields.io/badge/Paper-AAAI%202026-blue)](./EASiST_AAAI26_Final.pdf)

This repository contains the code implementation of the AAAI 2026 paper:  
**Efficient and Adaptive Simultaneous Speech Translation with Fully Unidirectional Architecture**.

## 📚 Table of Contents

- [1. Method Overview](#1-method-overview)
- [2. Repository Structure](#2-repository-structure)
- [3. Environment Setup](#3-environment-setup)
- [4. Data Preparation](#4-data-preparation)
- [5. Training Pipeline (Recommended Order)](#5-training-pipeline-recommended-order)
- [6. Inference and Evaluation](#6-inference-and-evaluation)
- [Citation](#citation)

---

## 1. Method Overview

EASiST has three core components:

1. **Fully unidirectional architecture**
   - Streaming speech encoder (`wav2vec-S`).
   - Autoregressive LLM decoding with cache.
   - Reusable cache on both speech and LLM sides to reduce recomputation.

2. **Interleaved SimulST formulation**
   - Explicit read/write control tokens:
     - `<|end-of-read|>`: switch from reading speech to writing translation.
     - `<|end-of-write|>`: switch from writing translation back to reading.
   - SimulST is trained as an interleaved autoregressive generation task.

3. **Three-stage training**
   - **Stage I (SimulMT pre-training)**: teach the LLM the interleaved format.
   - **Stage II (Offline ST alignment)**: align speech representations with LLM space.
   - **Stage III (Multi-task SFT)**: jointly optimize SimulST translation and policy behavior.

---

## 2. Repository Structure

- `easist/`
  - `train_easist_st.py`: Stage II/III training entry.
  - `offline_eval.py`: Offline ST evaluation.
  - `simul_eval.py`: SimulST inference and latency evaluation.
  - `latency_eval.py`: LAAL / LAAL-CA utilities.
  - `src/`: model, config, and dataset processing code.
- `scripts/train/`
  - `train_easist_stage1_simulmt.sh`
  - `train_easist_stage2_offlinest.sh`
  - `train_easist_stage3_simulst.sh`
- `scripts/eval/`
  - `offline_eval.sh`
  - `simul_eval.sh`
- `scripts/preprocess/`
  - `prep_mustc.sh`
  - `build_speech_seg_size.sh`
- `data/`
  - `offline_st/`, `simul_st/`, `simul_mt/`
  - Since MuST-C is no longer publicly available, only sample data is provided for pipeline debugging.

---

## 3. Environment Setup

### 3.1 Python dependencies

```bash
conda create -n easist python=3.10 -y
conda activate easist
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2 Extra dependency for Stage I

`scripts/train/train_easist_stage1_simulmt.sh` uses `llamafactory-cli train`, so you also need to install and configure [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), including dataset registration.

---

## 4. Data Preparation

### 4.1 MuST-C preprocessing (offline ST)

Script: `scripts/preprocess/prep_mustc.sh`

```bash
bash scripts/preprocess/prep_mustc.sh
```

Before running, update:
- `MUSTC_ROOT=/path/to/MuST-C`
- `OUTPUT_PATH=data/offline_st`

### 4.2 Data format

#### A) Stage I / SimulMT samples (`data/simul_mt/*.json`)

These samples can be constructed using prompt templates in `scripts/preprocess/data_curation_prompt.py`.

In text SFT data, the `output` field includes:
- `<|end-of-read|>`
- `<|end-of-write|>`

#### B) Stage II / Offline ST samples (`data/offline_st/*.json`)

Each sample includes:
- `audio`: `/path/to/audio.wav:start:frames`
- `src_text` / `tgt_text`
- `src_lang` / `tgt_lang`

#### C) Stage III / SimulST samples (`data/simul_st/*.json`)

Additional fields include:
- `chunk_src` / `chunk_tgt` / `chunk_hypo`
- `chunk_seg_time`: chunk-level time boundaries obtained via [MFA (Montreal Forced Aligner)](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner).
- `speech_seg_size`: cumulative audio sample boundaries per chunk, used for streaming read/write switching.
  - Processing script: `scripts/preprocess/build_speech_seg_size.sh`
- `data_type`: `streaming` or `offline`

---

## 5. Training Pipeline (Recommended Order)

```bash
git clone https://github.com/biaofuxmu/EASiST
cd EASiST
```

### 5.1 Stage I: SimulMT pre-training (text)

```bash
bash scripts/train/train_easist_stage1_simulmt.sh
```

Purpose:
- teach the LLM the interleaved read/write format;
- produce a Stage I checkpoint for later speech alignment.

Adjust the following in the script based on your environment:
- `base_model_path`
- `save_path`
- dataset name mapping (LLaMA-Factory registration)
- special token addition in Stage I: `<|end-of-read|>,<|end-of-write|>`
  - Reference: [EAST implementation](https://github.com/biaofuxmu/EAST/blob/main/src/llamafactory/train/sft/workflow.py#L33) (adds tokens via `tokenizer.add_special_tokens`)
  - Reference: [Latest LlamaFactory `model_args.py`](https://github.com/hiyouga/LlamaFactory/blob/main/src/llamafactory/hparams/model_args.py#L72) (adds tokens via argument-based config)

### 5.2 Stage II: Offline ST modality alignment

```bash
bash scripts/train/train_easist_stage2_offlinest.sh
```

Defaults:
- freeze LLM: `--frozen_modules "llm"`
- task type: `--task_type offline_st`

Update:
- `llm_model` (typically Stage I output)
- `speech_model` (e.g., wav2vec-S checkpoint)
- `save_path`

### 5.3 Stage III: SimulST multi-task SFT

```bash
bash scripts/train/train_easist_stage3_simulst.sh
```

Defaults:
- task type: `--task_type simul_st`
- enable policy supervision: `--speech_label True --text_label True`
- keep LLM frozen; train speech side / adapter / policy head

Update:
- `speech_llm_model` (typically Stage II output)
- `save_path`

---

## 6. Inference and Evaluation

### 6.1 Offline ST evaluation

```bash
bash scripts/eval/offline_eval.sh
```

Outputs:
- `prediction.json`
- `results.json` (BLEU and speed statistics)

### 6.2 SimulST evaluation

```bash
bash scripts/eval/simul_eval.sh
```

The script sweeps `latency_prob` from `0.1` to `0.6`. Each point generates a result directory containing:
- `prediction.json`
- `results.json` (`BLEU`, `LAAL`, `LAAL_CA`)

---

## Citation

If this project is useful for your research, please cite:

```bibtex
@article{fu2026easist,
  title={Efficient and Adaptive Simultaneous Speech Translation with Fully Unidirectional Architecture},
  author={Fu, Biao and Yu, Donglei and Liao, Minpeng and Li, Chengxi and Chen, Xinjie and Chen, Yidong and Fan, Kai and Shi, Xiaodong},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```
