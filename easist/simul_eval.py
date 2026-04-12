import os
import sys
import json
from datasets import load_dataset
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm

import sacrebleu
from statistics import mean
from transformers import AutoTokenizer, AutoFeatureExtractor, DynamicCache
from src.modeling_speech_llama import SpeechLlamaModel
from src.configuration_speech_llama import SpeechLlamaConfig
from src.speech_to_text_paired_dataset import get_waveform
import soundfile as sf

import time
from latency_eval import compute_delays, LengthAdaptiveAverageLagging

class SimulInference:
    def __init__(self, args):
        self.args = args
        self.test_data = self.load_eval_datasets(self.args.data_path)
        self.load_model(self.args.speech_llama)

        self.gen_kwargs = self.prepare_gen_config(args)
        self.set_special_tokens()
        self.set_encoder_config()
        
        self.instruction = self.args.instruction.encode().decode('unicode_escape')
        self.latency_prob = args.latency_prob

        self.predictions = []
        self.exceptions = []

    def load_model(self, model_path):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(model_path))
        try:
            self.extractor = AutoFeatureExtractor.from_pretrained(model_path)
        except:
            self.extractor = AutoFeatureExtractor.from_pretrained(os.path.dirname(model_path))

        speech_llama_config = SpeechLlamaConfig.from_pretrained(model_path)
        self.model = SpeechLlamaModel.from_pretrained(
            pretrained_model_name_or_path=model_path,
            config=speech_llama_config
        )
        self.model.to(torch.bfloat16).cuda()
        self.model.eval()

    def load_eval_datasets(self, data_path):
        with open(data_path, "r") as f:
            data = json.load(f)
        return data

    def prepare_gen_config(self, args):
        gen_kwargs = {}
        gen_kwargs["max_new_tokens"] = args.max_new_tokens
        gen_kwargs["min_new_tokens"] = args.min_new_tokens
        gen_kwargs["do_sample"] = args.do_sample
        gen_kwargs["num_beams"] = args.num_beams
        gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        return gen_kwargs

    def set_special_tokens(self):
        speech_eos_tok = "<|end-of-read|>"
        st_eos_tok = "<|end-of-write|>"
        eos_tok = "<|eot_id|>"

        self.speech_eos_tok_id = self.tokenizer(speech_eos_tok, add_special_tokens=False).input_ids[0]
        self.st_eos_tok_id = self.tokenizer(st_eos_tok, add_special_tokens=False).input_ids[0]
        self.eos_tok_id = self.tokenizer(eos_tok, add_special_tokens=False).input_ids[0]

    def set_encoder_config(self):
        self.speech_segment_size = 400

        self.main_context = self.model.speech_model.encoder.main_context
        self.right_context = self.model.speech_model.encoder.right_context

        self.block_size = self.main_context + self.right_context
        self.step_frames = self.main_context 
        
        print(f"main context: {self.main_context}\tright context: {self.right_context}")


    def extra_cnn_features(self, segment, sampling_rate):
        block_inputs = self.extractor(segment, sampling_rate=sampling_rate, return_tensors="pt").to(torch.bfloat16).to('cuda')
        cnn_features = self.model.speech_model.extract_cnn_features(block_inputs.input_values)
        return cnn_features


    def get_speech_embeds(self, cnn_features, past_speech_key_values, processed_frames):
        start_frame = processed_frames
        end_frame = processed_frames + self.block_size
        block_outputs = self.model.speech_model.forward_encoder(
            cnn_features[:,:end_frame], 
            past_key_values=past_speech_key_values, 
            use_cache=True
        )

        speech_embeds = block_outputs.last_hidden_state
        speech_embeds = self.model.adapter(speech_embeds)

        past_speech_key_values = block_outputs.past_key_values

        return speech_embeds, past_speech_key_values


    def cal_write_prob(self, inputs_embeds, past_key_values):
        self.gen_kwargs['num_beams'] = 1
        self.gen_kwargs["max_new_tokens"] = 1
        self.gen_kwargs["eos_token_id"] = [self.st_eos_tok_id]

        output_content = self.model.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones((1, inputs_embeds.shape[1])).to(inputs_embeds.device),
            output_hidden_states=True,
            return_dict_in_generate=True,
            past_key_values=past_key_values,
            **self.gen_kwargs
        )

        past_key_values = output_content.past_key_values

        rw_logits = self.model.decision_head(output_content.hidden_states[0][-1])
        rw_probs = torch.softmax(rw_logits[0][-1], dim=0)
        write_prob = rw_probs[1].item()

        return write_prob, past_key_values

    def step_elapsed(self, start, end, length):
        if length <= 1:
            return [end]
    
        step = (end - start) / length
        elapsed = [start + i * step for i in range(length+1)]
        
        return elapsed[1:]

    def write_step(self, inputs_embeds, past_key_values, start_time):
        self.gen_kwargs["max_new_tokens"] = self.args.max_new_tokens
        self.gen_kwargs["num_beams"] = self.args.num_beams
        self.gen_kwargs["suppress_tokens"] = [self.speech_eos_tok_id]

        action_embed = self.model.llama_model.get_input_embeddings()(torch.tensor([self.speech_eos_tok_id]).to('cuda')).unsqueeze(0)
        inputs_embeds = torch.cat((inputs_embeds, action_embed), dim=1)

        input_length = inputs_embeds.shape[1]

        start_gen_time = (time.time() - start_time) * 1000
        model_output = self.model.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones((1, input_length)).to(inputs_embeds.device),
            use_cache=True,
            return_dict_in_generate=True,
            past_key_values=past_key_values,
            **self.gen_kwargs
        )
        end_gen_time = (time.time() - start_time) * 1000
        past_key_values = model_output.past_key_values
        generated_ids = model_output.sequences

        generated_embeds = self.model.llama_model.get_input_embeddings()(generated_ids.to(inputs_embeds.device))
        inputs_embeds = torch.cat((inputs_embeds, generated_embeds), dim=1)

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        pred = response.replace("<|end-of-write|>", "").replace("<|eot_id|>", "")

        elapsed = self.step_elapsed(start_gen_time, end_gen_time, len(pred.split()))

        return pred, inputs_embeds, past_key_values, elapsed


    def eval_instance(self, index, sample):
        wav_file = sample['audio']

        instruction = self.instruction.format(src_lang=sample["src_lang"], tgt_lang=sample["tgt_lang"])

        wav_file, start, frame = wav_file.split(":")
        waveform, sampling_rate = sf.read(wav_file, dtype="float32",always_2d=True, frames=int(frame), start=int(start))
        waveform = waveform.T

        step = 1
        decision_step = 0
        segment_size = self.speech_segment_size * step
        processed_frames = 0
        
        past_key_values = DynamicCache()
        past_speech_key_values = DynamicCache()
        finish_read = False
        inputs_embeds = None

        preds = []
        elapseds = []
        delays = []
        write_probs = []

        try:
            start_time = time.time()
            while segment_size <= len(waveform[0]):
                if segment_size >= len(waveform[0]):
                    finish_read = True

                if finish_read:
                    self.model.speech_model.encoder.right_context = 0
                else:
                    self.model.speech_model.encoder.right_context = self.right_context

                ## read 
                need_frames = (self.block_size + self.step_frames * decision_step) * 320

                if segment_size >= need_frames or finish_read:
                    segment = waveform[:,:segment_size]

                    cnn_features = self.extra_cnn_features(segment, sampling_rate)
                    current_frame = cnn_features.size(1)

                    if (processed_frames == 0 and current_frame >= self.block_size) or \
                        (processed_frames > 0 and current_frame - processed_frames >= self.block_size) \
                        or finish_read:

                        decision_step += 1
                        speech_embeds, past_speech_key_values = self.get_speech_embeds(
                            cnn_features, 
                            past_speech_key_values, 
                            processed_frames
                        )

                        processed_frames = past_speech_key_values.get_seq_length() if past_speech_key_values is not None else 0

                        if inputs_embeds is None and len(past_key_values) == 0:
                            tok_prompt_id =  self.tokenizer(instruction).input_ids
                            prompt_embed = self.model.llama_model.get_input_embeddings()(torch.tensor(tok_prompt_id).to('cuda')).unsqueeze(0)
                            inputs_embeds = torch.cat((prompt_embed, speech_embeds), dim=1)
                        else:
                            inputs_embeds = torch.cat((inputs_embeds, speech_embeds),dim=1)

                        write_prob, past_key_values = self.cal_write_prob(inputs_embeds, past_key_values)

                        if write_prob > self.latency_prob or finish_read:
                            write_probs.append(write_prob)
                            delays.append(min(segment_size, len(waveform[0])) / 16)

                            pred, inputs_embeds, past_key_values, elapsed = self.write_step(inputs_embeds, past_key_values, start_time)

                            preds.append(pred)
                            elapseds.extend(elapsed)

                if finish_read:
                    break
                
                step += 1
                segment_size = min(self.speech_segment_size * step, len(waveform[0]))

                if segment_size >= len(waveform[0]):
                    finish_read = True

        except Exception as e:
            print(e)
            self.exceptions.append(
                {
                    "index": index,
                    "audio": sample['audio'],
                    "src_text": sample["src_text"],
                    "reference": sample["tgt_text"],
                    "duration": len(waveform[0]) / 16,
                   "src_lang": sample["src_lang"],
                    "tgt_lang": sample["tgt_lang"],
                    "hypo": preds,
                    "delays": str(delays),
                    "elapseds": str(elapseds),
                    "Exception": str(e)
                }
            ) 
            return

        predict = "".join(preds)

        self.predictions.append(
            {
                "index": index,
                "audio": sample['audio'],
                "src_text": sample["src_text"],
                "reference": sample["tgt_text"],
                "prediction": predict,
                "duration": len(waveform[0]) / 16,
                "src_lang": sample["src_lang"],
                "tgt_lang": sample["tgt_lang"],
                "hypo": preds,
                "delays": str(delays),
                "elapseds": str(elapseds),
                "write_probs": str(write_probs),
            }
        )


    def simul_eval(self):
        for index, sample in tqdm(enumerate(self.test_data), total=len(self.test_data)):
            # if index != 1:
                # continue

            with torch.no_grad():
                self.eval_instance(index, sample)

        self.model.cpu()
        torch.cuda.empty_cache()

        results = self.cal_scores()

        self.save_results(results, self.args.output_dir)

    def cal_scores(self):
        hypos = []
        refs = []
        results = {}
        LAALs = []
        LAAL_CAs = []
        self.errors = []
        self.error_formats = []

        for prediction in self.predictions:
            predict = prediction["prediction"]
            ref = prediction["reference"]
            src_lang = prediction["src_lang"]
            tgt_lang = prediction["tgt_lang"]
            hypo = prediction["hypo"]
            src_len = prediction["duration"]
            delay = json.loads(prediction["delays"])
            elapsed = json.loads(prediction["elapseds"])

            hypos.append(predict)
            refs.append(ref)

            pred_len = len(predict.split())
            ref_len = len(ref.split())

            bleu = sacrebleu.sentence_bleu(predict, [ref], tokenize="13a").score
            delays, elapseds = [], []
            LAAL = None
            LAAL_CA = None

            try:
                delays, elapseds = compute_delays(delay, hypo, elapsed, src_lang, tgt_lang)
                assert len(delays) == len(elapseds)
                LAAL = LengthAdaptiveAverageLagging(delays, src_len, ref_len)
                LAAL_CA = LengthAdaptiveAverageLagging(elapseds, src_len, ref_len)

                LAALs.append(LAAL)
                LAAL_CAs.append(LAAL_CA)
            except Exception as e:
                print(e)
                self.errors.append(
                    {
                        "index": prediction["index"],
                        "exception": str(e),
                        "reference": ref,
                        "prediction": predict,
                        "BLEU": bleu,
                    }
                )

            prediction["BLEU"] = bleu
            prediction["LAAL"] = LAAL
            prediction["LAAL_CA"] = LAAL_CA
            prediction["source_length"] = src_len
            prediction["reference_length"] = ref_len
            prediction["prediction_length"] = pred_len
            prediction["delays"] = str(delays)
            prediction["elapseds"] = str(elapseds)

        bleu_score = sacrebleu.corpus_bleu(
                        hypos,
                        [refs],
                        tokenize="13a"
                    ).score

        results["BLEU"] = bleu_score
        results["LAAL"] = mean(LAALs) if len(LAALs) > 0 else None
        results["LAAL_CA"] = mean(LAAL_CAs) if len(LAAL_CAs) > 0 else None

        print(len(hypos), len(LAALs), len(self.predictions))

        return results


    def save_results(self, results, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prediction_file = os.path.join(output_dir, "prediction.json")
        with open(prediction_file, "w", encoding="utf8") as f:
            json.dump(self.predictions, f, ensure_ascii=False, indent=4)

        result_file = os.path.join(output_dir, "results.json")
        with open(result_file, "w", encoding="utf8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        if len(self.errors) > 0:
            error_file = os.path.join(output_dir, "errors.json")
            with open(error_file, "w", encoding="utf8") as f:
                json.dump(self.errors, f, ensure_ascii=False, indent=4)

        if len(self.exceptions) > 0:
            exceptions_file = os.path.join(output_dir, "exceptions.json")
            with open(exceptions_file, "w", encoding="utf8") as f:
                json.dump(self.exceptions, f, ensure_ascii=False, indent=4)


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Path to the input file", required=True
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Path to the output file", required=True
    )
    parser.add_argument(
        "--speech_llama", type=str, default=None,
        help="Path to the blsp model", required=True
    )
    parser.add_argument(
        "--instruction", type=str, default="",
        help="the general instruction for each example"
    )
    ### args for generation
    parser.add_argument(
        "--max_new_tokens", type=int, default=128,
        help="max new tokens for generation"
    )
    parser.add_argument(
        "--min_new_tokens", type=int, default=1,
        help="min new tokens for generation"
    )
    parser.add_argument(
        "--do_sample", action="store_true",
        help="whether do sample. For ST task, we will use greedy search to ensure stable output"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="temperature for generation"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.75,
        help="top_p for generation"
    )
    parser.add_argument(
        "--num_beams", type=int, default=5,
        help="number of beams of beam search"
    )
    parser.add_argument(
        "--latency_prob", type=float, default=0.9,
        help="latency prob"
    )
    args = parser.parse_args()

    return args

def main():

    args = load_args()
    infer = SimulInference(args)
    infer.simul_eval()

if __name__=='__main__':
    main()
