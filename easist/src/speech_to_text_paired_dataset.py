import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, BinaryIO
import fire
import soundfile as sf

import numpy as np
import torch
import random
import datasets
from dataclasses import dataclass
from itertools import accumulate, groupby
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
# import pdb;pdb.set_trace()

logger = logging.getLogger(__name__)


def interleave_tgt_text(
    chunk_tgt,
    tgt_lang,
    speech_eos_tok="<|end-of-read|>",
    st_eos_tok="<|end-of-write|>",
):
    output = ""
    for idx, tgt in enumerate(chunk_tgt):
        tgt_m = tgt.strip()
        tgt_seg = "" if tgt_lang == "Chinese" or tgt_m == "" else " "
        if idx > 0:
            tgt_m = f"{tgt_seg}{tgt_m}"
        output = output + f"{speech_eos_tok}{tgt_m}{st_eos_tok}"
    return output

def split_text_segs(input_ids, seg_id):
    groups = list(accumulate(1 if x == seg_id else 0 for x in input_ids))
    sizes = [len(list(group)) for _, group in groupby(groups)]
    
    return sizes

def process_streaming_dataset(batch, tokenizer, instruction_prefix, instruction_suffix, task_type="simul_st"):
    speech_eos_tok = "<|end-of-read|>"
    st_eos_tok = "<|end-of-write|>"

    speech_eos_tok_id = tokenizer(speech_eos_tok,add_special_tokens=False).input_ids
    if len(speech_eos_tok_id) != 1:
        raise ValueError(f"Expected {speech_eos_tok} to map to one token, got ids={speech_eos_tok_id}")
    if task_type != "simul_st":
        raise ValueError(f"Only 'simul_st' is supported, but got '{task_type}'.")
    
    instruction_prefix = instruction_prefix.encode().decode('unicode_escape')
    instruction_suffix = instruction_suffix.encode().decode('unicode_escape')
    try:
        instruction_prefix_ = instruction_prefix.format(src_lang=batch['src_lang'], tgt_lang=batch['tgt_lang'], latency=batch['latency-level'])
    except Exception:
        instruction_prefix_ = instruction_prefix.format(src_lang=batch['src_lang'], tgt_lang=batch['tgt_lang'])

    audio_path = batch["audio"]

    try:
        info = sf.info(audio_path.split(":")[0])
        is_readable = True
    except:
        is_readable = False

    # User + Assistant
    instruction = instruction_prefix_ + instruction_suffix
    input_ids = tokenizer(instruction, add_special_tokens=False).input_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(input_ids)

    response = interleave_tgt_text(
        chunk_tgt=batch['chunk_tgt'],
        tgt_lang=batch['tgt_lang'],
        speech_eos_tok=speech_eos_tok,
        st_eos_tok=st_eos_tok
    )

    suffix_input_ids = tokenizer(response, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    suffix_attention_mask = [1] * len(suffix_input_ids)
    suffix_labels = suffix_input_ids
    text_seg_size = split_text_segs(suffix_input_ids, speech_eos_tok_id[0])
    
    if batch["data_type"] == "streaming":
        is_streaming = True
    elif batch["data_type"] == "offline":
        is_streaming = False
    else:
        raise ValueError(f"Unknown data_type: {batch.get('data_type')}")

    sample = {}
    sample["input_ids"] = input_ids
    sample["attention_mask"] = attention_mask
    sample["labels"] = labels
    sample["suffix_input_ids"] = suffix_input_ids
    sample["suffix_attention_mask"] = suffix_attention_mask
    sample["suffix_labels"] = suffix_labels
    sample["audio_path"] = audio_path
    sample["is_readable"] = is_readable

    sample["chunk_seg_time"] = batch["speech_seg_size"] 
    sample["text_seg_size"] = text_seg_size
    sample["is_streaming"] = is_streaming

    assert len(sample["chunk_seg_time"]) == len(sample["text_seg_size"])

    return sample

def process_dataset(batch, tokenizer, instruction_prefix, instruction_suffix, task_type="offline_st"):
    if "<|end-of-read|>" not in tokenizer.special_tokens_map and "<|end-of-write|>" not in tokenizer.special_tokens_map:
        st_eos_token = ""
    else:
        st_eos_token = "<|end-of-write|>"

    instruction_prefix = instruction_prefix.encode().decode('unicode_escape')
    instruction_suffix = instruction_suffix.encode().decode('unicode_escape')
    if task_type != "offline_st":
        raise ValueError(f"Only 'offline_st' is supported, but got '{task_type}'.")
    instruction_prefix_ = instruction_prefix.format(src_lang=batch['src_lang'], tgt_lang=batch['tgt_lang'])
    input_ids = tokenizer(instruction_prefix_, add_special_tokens=False).input_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(input_ids)

    audio_path = batch["audio"]
    try:
        info = sf.info(audio_path.split(":")[0])
        is_readable = True
    except:
        is_readable = False

    suffix_input_ids, suffix_attention_mask, suffix_labels = [], [], []

    ### Assistant
    new_input_ids = tokenizer(instruction_suffix, add_special_tokens=False).input_ids
    suffix_input_ids += new_input_ids
    suffix_attention_mask += [1] * len(new_input_ids)
    suffix_labels += [-100] * len(new_input_ids)
    response = batch["tgt_text"] + st_eos_token

    new_input_ids = tokenizer(response, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    suffix_input_ids += new_input_ids
    suffix_attention_mask += [1] * len(new_input_ids)
    suffix_labels += new_input_ids

    sample = {}
    sample["input_ids"] = input_ids
    sample["attention_mask"] = attention_mask
    sample["labels"] = labels
    sample["suffix_input_ids"] = suffix_input_ids
    sample["suffix_attention_mask"] = suffix_attention_mask
    sample["suffix_labels"] = suffix_labels
    sample["audio_path"] = audio_path
    sample["is_readable"] = is_readable

    return sample

def load_speech_to_text_paired_dataset(
    dataroot="",
    manifest_files="",
    tokenizer=None,
    instruction_prefix="",
    instruction_suffix="",
    speech_model_type="wav2vec_s",
    task_type="offline_st",
    num_proc=32,
):
    logger.warning(f"load dataset from scratch from {dataroot}/{manifest_files}")
    manifest_files_list = manifest_files.split(",")

    raw_dataset = datasets.load_dataset(
        dataroot, data_files=manifest_files_list, split="train", streaming=False
    )

    if task_type == "simul_st":
        process_dataset_func = process_streaming_dataset
    elif task_type == "offline_st":
        process_dataset_func = process_dataset
    else:
        raise ValueError(f"The data processing function do not support the '{task_type}' task!")

    dataset = raw_dataset.map(
        process_dataset_func,
        fn_kwargs={
            "tokenizer": tokenizer,
            "instruction_prefix": instruction_prefix,
            "instruction_suffix": instruction_suffix,
            "task_type": task_type,
        },
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=False,
        num_proc=num_proc,
    )

    def is_readable(flag):
        return flag

    dataset = dataset.filter(
        is_readable,
        input_columns=["is_readable"]
    )

    return dataset


def collate_tokens(
    values: List[List[int]],
    pad_id: int
):
    size = max(len(v) for v in values)
    batch_size = len(values)
    res = torch.LongTensor(batch_size, size).fill_(pad_id)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(torch.LongTensor(v), res[i][: len(v)])

    return res

def collate_segs(
    values: List[List[float]],
    sampling_rate: int ,
    pad_id: int
):
    size = max(len(v) for v in values)
    batch_size = len(values)
    res = torch.LongTensor(batch_size, size).fill_(pad_id)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(torch.LongTensor(v), res[i][: len(v)])

    return res

def get_waveform(
    path_or_fp: Union[str, BinaryIO],
    normalization=True,
    mono=True,
    frames=-1,
    start=0,
    always_2d=False,
    output_sample_rate=16000,
) -> Tuple[np.ndarray, int]:
    meta = path_or_fp.split(":")
    if len(meta) == 3 and (meta[0].endswith(".wav") or meta[0].endswith(".flac")):
        path_or_fp = meta[0]
        start = int(meta[1])
        frames = int(meta[2])
    else:
        path_or_fp = path_or_fp

    if isinstance(path_or_fp, str):
        ext = Path(path_or_fp).suffix
        if ext in [".wav", ".flac", ".ogg", ".mp3"]:
            pass
        else:
            raise ValueError(f"Unsupported audio format: {ext}")

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile to load WAV/FLACC/OGG/MP3 audios")
    waveform, sample_rate = sf.read(
        path_or_fp, dtype="float32", always_2d=True, frames=frames, start=start
    )
    waveform = waveform.T

    waveform, sample_rate = convert_waveform(waveform, sample_rate, to_mono=mono, to_sample_rate=output_sample_rate)
    if not normalization:
        waveform *= 2 ** 15
    if not always_2d:
        waveform = waveform.squeeze(axis=0)
    return waveform

def convert_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    normalize_volume: bool = False,
    to_mono: bool = False,
    to_sample_rate: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    """convert a waveform:
    - to a target sample rate
    - from multi-channel to mono channel
    - volume normalization
    Args:
        waveform (numpy.ndarray or torch.Tensor): 2D original waveform
            (channels x length)
        sample_rate (int): original sample rate
        normalize_volume (bool): perform volume normalization
        to_mono (bool): convert to mono channel if having multiple channels
        to_sample_rate (Optional[int]): target sample rate
    Returns:
        waveform (numpy.ndarray): converted 2D waveform (channels x length)
        sample_rate (float): target sample rate
    """
    try:
        import torchaudio.sox_effects as ta_sox
    except ImportError:
        raise ImportError("Please install torchaudio: pip install torchaudio")

    effects = []
    if normalize_volume:
        effects.append(["gain", "-n"])
    if to_sample_rate is not None and to_sample_rate != sample_rate:
        effects.append(["rate", f"{to_sample_rate}"])
    if to_mono and waveform.shape[0] > 1:
        effects.append(["channels", "1"])
    if len(effects) > 0:
        is_np_input = isinstance(waveform, np.ndarray)
        _waveform = torch.from_numpy(waveform) if is_np_input else waveform
        converted, converted_sample_rate = ta_sox.apply_effects_tensor(
            _waveform, sample_rate, effects
        )
        if is_np_input:
            converted = converted.numpy()
        return converted, converted_sample_rate
    return waveform, sample_rate


@dataclass
class SpeechToTextPairedDataCollator:
    """
    Data collator that will dynamically pad the inputs received.
    """
    pad_id: int = 0
    sampling_rate: int = 16000
    extractor: Wav2Vec2FeatureExtractor = Wav2Vec2FeatureExtractor()
    speech_model_type: str = "wav2vec_s"

    def __call__(self, samples: List[Dict]):
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [sample["attention_mask"] for sample in samples]
        labels = [sample["labels"] for sample in samples]
        suffix_input_ids = [sample["suffix_input_ids"] for sample in samples]
        suffix_attention_mask = [sample["suffix_attention_mask"] for sample in samples]
        suffix_labels = [sample["suffix_labels"] for sample in samples]

        input_ids = collate_tokens(input_ids, self.pad_id)
        attention_mask = collate_tokens(attention_mask, 0)
        labels = collate_tokens(labels, -100)
        suffix_input_ids = collate_tokens(suffix_input_ids, self.pad_id)
        suffix_attention_mask = collate_tokens(suffix_attention_mask, 0)
        suffix_labels = collate_tokens(suffix_labels, -100)

        raw_speech = [
            get_waveform(sample["audio_path"], output_sample_rate=self.sampling_rate) for sample in samples
        ]
        speech_inputs = self.extractor(
            raw_speech, 
            sampling_rate=self.sampling_rate, 
            return_attention_mask=True,
            return_tensors="pt",
            padding=True
        )
        
        chunk_seg_time = [sample["chunk_seg_time"] for sample in samples] if "chunk_seg_time" in samples[0] else None
        chunk_seg_time = collate_segs(chunk_seg_time, self.sampling_rate, -1) if chunk_seg_time is not None else None

        text_seg_sizes = [sample["text_seg_size"] for sample in samples] if "text_seg_size" in samples[0] else None
        text_seg_sizes = collate_tokens(text_seg_sizes, -1) if chunk_seg_time is not None else None
        
        is_streamings = [sample["is_streaming"] for sample in samples] if "is_streaming" in samples[0] else None
        # is_streamings = collate_tokens(is_streamings, -1) if is_streamings is not None else None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "suffix_input_ids": suffix_input_ids,
            "suffix_attention_mask": suffix_attention_mask,
            "suffix_labels": suffix_labels,
            "speech_inputs": speech_inputs,
            "chunk_seg_time": chunk_seg_time,
            "text_seg_sizes": text_seg_sizes,
            "is_streamings": is_streamings,
        }


def offline_process(
    data="",
    manifest_files="",
    lm_path="",
    instruction_prefix="",
    instruction_suffix="",
    num_proc=8,
    task_type="offline_st",
    speech_model_type="wav2vec_s",
):
    text_tokenizer = AutoTokenizer.from_pretrained(lm_path)

    dataset = load_speech_to_text_paired_dataset(
        dataroot=data,
        manifest_files=manifest_files,
        tokenizer=text_tokenizer,
        instruction_prefix=instruction_prefix,
        instruction_suffix=instruction_suffix,
        speech_model_type=speech_model_type,
        task_type=task_type,
        num_proc=num_proc,
    )
    for key in dataset[0].keys():
        if key != "audio_path" and key != "is_readable" and key != "is_streaming":
            print(key, len(dataset[0][key]))
        else:
            print(key, dataset[0][key])
    print(len(dataset))


if __name__ == "__main__":
    fire.Fire({
        "offline": offline_process,
    })