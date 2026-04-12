import argparse
import bisect
import json

import soundfile as sf
from tqdm import tqdm
from transformers import AutoConfig

WAV2VEC2_DEFAULT_CONV_KERNEL = (10, 3, 3, 3, 3, 2, 2)
WAV2VEC2_DEFAULT_CONV_STRIDE = (5, 2, 2, 2, 2, 2, 2)


def find_greater_closest_values(chunk_seg_time, segment_sizes):
    result = []
    for time in chunk_seg_time:
        idx = bisect.bisect_left(segment_sizes, time)
        if idx >= len(segment_sizes):
            idx = len(segment_sizes) - 1
        result.append(segment_sizes[idx])
    return result


def get_feat_extract_output_lengths(input_length, conv_kernel, conv_stride):
    output_length = int(input_length)
    for kernel, stride in zip(conv_kernel, conv_stride):
        output_length = (output_length - kernel) // stride + 1
        if output_length <= 0:
            return 0
    return output_length


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate speech_seg_size for SimulST data from chunk_seg_time."
    )
    parser.add_argument(
        "--speech_model_path",
        type=str,
        default="facebook/wav2vec2-base-960h",
        help="Pretrained speech model path for loading conv config only.",
    )
    parser.add_argument(
        "--simul_path",
        type=str,
        required=True,
        help="Input SimulST json file path.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output json file path.",
    )
    parser.add_argument(
        "--main_context",
        type=int,
        default=32,
        help="Main context used in streaming encoder logic.",
    )
    parser.add_argument(
        "--right_context",
        type=int,
        default=16,
        help="Right context used in streaming encoder logic.",
    )
    parser.add_argument(
        "--speech_segment_size",
        type=int,
        default=400,
        help="Segment size multiplier used by preprocessing loop.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    speech_config = AutoConfig.from_pretrained(args.speech_model_path)
    conv_kernel = getattr(speech_config, "conv_kernel", None)
    conv_stride = getattr(speech_config, "conv_stride", None)
    if conv_kernel is None or conv_stride is None:
        conv_kernel = WAV2VEC2_DEFAULT_CONV_KERNEL
        conv_stride = WAV2VEC2_DEFAULT_CONV_STRIDE
        print(
            "conv_kernel/conv_stride not found in config, "
            "fallback to wav2vec2.0 defaults."
        )

    main_context = args.main_context
    right_context = args.right_context
    speech_segment_size = args.speech_segment_size

    block_size = main_context + right_context

    print(f"main context: {main_context}\tright context: {right_context}")
    print(f"speech model: {args.speech_model_path}")
    print(f"conv_kernel: {conv_kernel}\tconv_stride: {conv_stride}")

    with open(args.simul_path, "r", encoding="utf8") as f:
        items = json.load(f)

    for item in tqdm(items, desc="processing"):
        audio = item["audio"]
        wav_file, start, frame = audio.split(":")
        waveform, sampling_rate = sf.read(
            wav_file,
            dtype="float32",
            always_2d=True,
            frames=int(frame),
            start=int(start),
        )
        waveform = waveform.T

        chunk_seg_time = [int(float(t) * sampling_rate) for t in item["chunk_seg_time"]]

        step = 1
        segment_size = speech_segment_size * step
        processed_frames = 0
        finish_read = False

        segment_sizes = []

        while segment_size <= len(waveform[0]):
            segment = waveform[:, :segment_size]
            current_frame = get_feat_extract_output_lengths(
                input_length=len(segment[0]),
                conv_kernel=conv_kernel,
                conv_stride=conv_stride,
            )

            if (
                (processed_frames == 0 and current_frame >= block_size)
                or (processed_frames > 0 and current_frame - processed_frames >= block_size)
                or finish_read
            ):
                segment_sizes.append(segment_size)

                if not finish_read:
                    processed_frames = int(current_frame / block_size) * block_size
                else:
                    processed_frames = current_frame

            if finish_read:
                break

            step += 1
            segment_size = min(speech_segment_size * step, len(waveform[0]))
            if segment_size == len(waveform[0]):
                finish_read = True

        # Guard against extremely short audio where no segment boundary is emitted.
        if len(segment_sizes) == 0:
            segment_sizes = [len(waveform[0])]

        speech_seg_size = find_greater_closest_values(
            chunk_seg_time=sorted(chunk_seg_time),
            segment_sizes=sorted(segment_sizes),
        )

        assert len(chunk_seg_time) == len(speech_seg_size)
        item["speech_seg_size"] = speech_seg_size

    with open(args.output_path, "w", encoding="utf-8") as json_file:
        json.dump(items, json_file, ensure_ascii=False, indent=2)
    print(f"saved: {args.output_path}")


if __name__ == "__main__":
    main()
