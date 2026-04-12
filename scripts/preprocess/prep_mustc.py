#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple
import json
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

log = logging.getLogger(__name__)

class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["train", "dev", "tst-COMMON"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru", "ja", "zh"]
    LANG_TABLE = {
        "en": "English",
        "de": "German",
        "es": "Spanish",
        "fr": "French",
        "it": "Italian",
        "nl": "Dutch",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "ja": "Japanese",
        "zh": "Chinese",
    }

    def __init__(self, root: str, lang: str, split: str) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = Path(root) / f"en-{lang}" / "data" / split
        wav_root, txt_root = _root / "wav", _root / "txt"

        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load the MuST-C YAML files")
        with open(txt_root / f"{split}.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        for _lang in ["en", lang]:
            if _lang == "en":
                filename = txt_root / f"{split}.{_lang}"
            else:
                filename = txt_root / f"{split}.{_lang}"
            with open(filename) as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in tqdm(groupby(segments, lambda x: x["wav"])):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: float(x["offset"]))
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment["en"],
                        segment[lang],
                        segment["speaker_id"],
                        _id,
                    )
                )

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str, str, str, str]:
        return self.data[n]

    def __len__(self) -> int:
        return len(self.data)


def process(args, min_n_frames=1600, max_n_frames=480000):
    root = Path(args.data_root).absolute()
    lang = args.tgt_lang
    cur_root = root / f"en-{lang}"

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    # Extract features
    for split in MUSTC.SPLITS:
        is_train_split = split.startswith("train")
        is_test_split = split.startswith("tst-COMMON")
        manifest = []
        dataset = MUSTC(args.data_root, lang, split)
        for wav, offset, n_frames, sr, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
            if wav == "" or tgt_utt == "" or (not is_test_split and n_frames < min_n_frames) or (is_train_split and n_frames > max_n_frames):
                continue

            manifest.append(
                {
                    "id": utt_id,
                    "audio": f"{wav}:{offset}:{n_frames}",
                    "n_frames": n_frames,
                    "src_text": src_utt,
                    "src_lang": MUSTC.LANG_TABLE["en"],
                    "tgt_text": tgt_utt,
                    "tgt_lang": MUSTC.LANG_TABLE[lang],
                    "speaker": speaker_id,
                }
            )
        print(f"total: {len(dataset)}, remained: {len(manifest)}, filtered: {len(dataset) - len(manifest)}.")

        output_file = f"{output_path}/must-c.en-{lang}.{split}.json"
        with open(output_file, "w", encoding="utf8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--output-path", required=True, type=str)
    parser.add_argument("--tgt-lang", help="target language")
    args = parser.parse_args()

    MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "src_lang", "tgt_text", "tgt_lang", "speaker"]
    process(args)

if __name__ == "__main__":
    main()
