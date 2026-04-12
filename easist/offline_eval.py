import os
import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoFeatureExtractor
from src.modeling_speech_llama import SpeechLlamaModel
from src.configuration_speech_llama import SpeechLlamaConfig
from src.speech_to_text_paired_dataset import get_waveform

import torch
import sacrebleu
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, default=None,
        help="Path to the input file", required=True
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Path to the output file", required=True
    )
    parser.add_argument(
        "--speech_llama", type=str, default=None,
        help="Path to the SpeechLlama model", required=True
    )
    parser.add_argument(
        "--speech_model_type", type=str, default="whisper",
        help="type of speech model, whisper, wav2vec_s, wav2vecl"
    )
    parser.add_argument(
        "--instruction_prefix", type=str, default=None,
        help="The text prefix instruction before speech input, default None", required=True
    )
    parser.add_argument(
        "--instruction_suffix", type=str, default=None,
        help="The text suffix instruction after speech input, default None", required=True
    )
    ### args for generation
    parser.add_argument(
        "--max_new_tokens", type=int, default=128,
        help="max new tokens for generation"
    )
    parser.add_argument(
        "--num_beams", type=int, default=1,
        help="min new tokens for generation"
    )
    parser.add_argument(
        "--do_sample", action="store_true", default=False,
        help="whether do sample. For ST task, we will use greedy search to ensure stable output"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="temperature for generation"
    )

    args = parser.parse_args()

    instruction_prefix = args.instruction_prefix.encode().decode('unicode_escape')
    instruction_suffix = args.instruction_suffix.encode().decode('unicode_escape')

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.speech_llama)
    except:
        tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(args.speech_llama))
    try:
        extractor = AutoFeatureExtractor.from_pretrained(args.speech_llama)
    except:
        extractor = AutoFeatureExtractor.from_pretrained(os.path.dirname(args.speech_llama))


    gen_conf = {
        'num_beams': args.num_beams, 
        'max_new_tokens': args.max_new_tokens, 
        'eos_token_id': tokenizer.eos_token_id, 
        'pad_token_id': tokenizer.eos_token_id, 
        'do_sample': args.do_sample,
        'temperature': args.temperature
    }

    speech_llama_config = SpeechLlamaConfig.from_pretrained(args.speech_llama)
    model = SpeechLlamaModel.from_pretrained(
        pretrained_model_name_or_path=args.speech_llama,
        config=speech_llama_config
    )
    model.to(torch.bfloat16)
    model = model.cuda()
    model.eval()

    with open(args.input_file, "r") as fin:
        lines = json.load(fin)
    
    predictions = []
    results = {}
    hypos = []
    refs = []
    idx = 0
    start_gen_time = time.time()
    infer_token_nums = 0
    for data in tqdm(lines):

        src_lang = data['src_lang']
        tgt_lang = data['tgt_lang']

        # User
        instruction_prefix_ = instruction_prefix.format(src_lang=src_lang, tgt_lang=tgt_lang)
        input_ids = tokenizer(instruction_prefix_, return_tensors="pt", add_special_tokens=False)

        audio = data["audio"]
        speech = get_waveform(audio, output_sample_rate=extractor.sampling_rate) 

        speech_inputs = extractor(
            speech,
            sampling_rate=extractor.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt"
        )

        speech_inputs["input_values"] = speech_inputs.input_values.to(torch.bfloat16).cuda()
        speech_inputs["attention_mask"] = speech_inputs.attention_mask.cuda()

        suffix_input_ids = tokenizer(instruction_suffix, return_tensors="pt", add_special_tokens=False)

        output = model.generate(
            prefix_inputs=input_ids,
            suffix_inputs=suffix_input_ids,
            speech_inputs=speech_inputs,
            generation_config=gen_conf
        )
        infer_token_nums += len(output[0])

        response = tokenizer.decode(output[0], skip_special_tokens=True)

        hypos.append(response)

        refs.append(data["tgt_text"])

        if tgt_lang == "Chinese":
            tok = "zh"
        else:
            tok = "13a"

        bleu = sacrebleu.sentence_bleu(response, [data["tgt_text"]], tokenize=tok).score

        result = {
            "id": data["id"],
            "audio": audio,
            "n_frames": data["n_frames"],
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "src_text": data["src_text"],
            "reference": data["tgt_text"],
            "prediction": response,
            "BLEU": bleu
        }

        predictions.append(result)

    end_gen_time = time.time()

    if predictions[0]["tgt_lang"] == "Chinese":
        tok = "zh"
    else:
        tok = "13a"

    bleu_score = sacrebleu.corpus_bleu(
                    hypos,
                    [refs],
                    tokenize=tok
                ).score

    results["BLEU"] = bleu_score
    results["total_cost (s)"] = end_gen_time -  start_gen_time
    results["total_tokens"] = infer_token_nums
    results["time_per_token (ms)"] = (end_gen_time -  start_gen_time) / infer_token_nums * 1000

    print(results)

    prediction_file = os.path.join(args.output_path, "prediction.json")
    with open(prediction_file, "w", encoding="utf8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    result_file = os.path.join(args.output_path, "results.json")
    with open(result_file, "w", encoding="utf8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()