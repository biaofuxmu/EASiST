
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple

import datasets
import evaluate
import torch
import math
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, AutoFeatureExtractor

from src.speech_to_text_paired_dataset import load_speech_to_text_paired_dataset, SpeechToTextPairedDataCollator
from src.modeling_speech_llama import SpeechLlamaModel
from src.configuration_speech_llama import SpeechLlamaConfig
from src.modeling_speech_model import SpeechModels, SpeechConfigs

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    speech_llama: str = field(
        default="", metadata={"help": "Path to a pretrained SpeechLlama checkpoint. If provided, this checkpoint is loaded directly."}
    )
    llama_model: str = field(
        default="meta-llama/Llama-3.1-8B-Instruct", metadata={"help": "Path to the base LLM checkpoint used when `speech_llama` is empty."}
    )
    speech_model: str = field(
        default="facebook/wav2vec2-base-960h", metadata={"help": "Path to the speech encoder checkpoint used when `speech_llama` is empty."}
    )
    speech_model_type: str = field(
        default="wav2vec2", metadata={"help": "Speech encoder type. Supported values: `wav2vec_s`, `wav2vec2`."}
    )
    adapter_type: str = field(
        default="ffn", metadata={"help": "Adapter type between speech encoder and LLM. Supported values: `ffn`, `conv`, `blockconv`."}
    )
    adapter_inner_dim: int = field(
        default=2048, metadata={"help": "Hidden dimension used inside the adapter."}
    )
    conv_kernel_sizes: str = field(
        default="5,5", metadata={"help": "Comma-separated kernel sizes for convolutional adapters, for example `5,5`."}
    )
    frozen_modules: str = field(
        default="", metadata={"help": "Comma-separated module names to freeze during training. Options include `speech`, `llm`, `adapter`."}
    )
    speech_label: bool = field(
        default=False, metadata={"help": "Enable auxiliary speech/read-write decision labels."}
    )
    text_label: bool = field(
        default=True, metadata={"help": "Use text-side labels for the auxiliary decision branch when enabled."}
    )
    cfd_weight: float = field(
        default=1.0, metadata={"help": "Loss weight of the auxiliary decision objective."}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data: str = field(
        metadata={
            "help": "Dataset root path used by `datasets.load_dataset`."
        },
    )
    manifest_files: str = field(
        default="",
        metadata={
            "help": "Comma-separated manifest file names for training data."
        },
    )
    eval_manifest_files: str = field(
        default=None,
        metadata={
            "help": "Optional comma-separated manifest file names for evaluation data."
        },
    )
    instruction_prefix: str = field(
        default="",
        metadata={
            "help": "Instruction prefix before content. Can include placeholders like `{src_lang}`, `{tgt_lang}`, and optional `{latency}`."
        },
    )
    instruction_suffix: str = field(
        default="",
        metadata={
            "help": "Instruction suffix appended after the prefix/context."
        },
    )
    task_type: str = field(
        default="offline_st", metadata={"help": "Task type. Supported values: `offline_st`, `simul_st`."}
    )

def _noisy_mean_initialization(embed_weight: "torch.Tensor", num_new_tokens: int) -> None:
    embedding_dim = embed_weight.size(1)
    avg_weight = embed_weight[:-num_new_tokens].mean(dim=0, keepdim=True)
    noise_weight = torch.empty_like(embed_weight[-num_new_tokens:])
    noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))
    embed_weight[-num_new_tokens:] = avg_weight + noise_weight

def resize_embedding_layer(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer") -> None:
    r"""
    Resize token embeddings.
    """
    from contextlib import nullcontext
    context_maybe_zero3 = nullcontext()

    with context_maybe_zero3:
        current_embedding_size = model.llama_model.get_input_embeddings().weight.size(0)

    if len(tokenizer) > current_embedding_size:
        if getattr(model.llama_model, "quantization_method", None):
            raise ValueError("Cannot resize embedding layers of a quantized model.")

        if not isinstance(model.llama_model.get_output_embeddings(), torch.nn.Linear):
            raise ValueError("Current model does not support resizing embedding layers.")

        model.llama_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        with context_maybe_zero3:
            new_embedding_size = model.llama_model.get_input_embeddings().weight.size(0)
            num_new_tokens = new_embedding_size - current_embedding_size
            _noisy_mean_initialization(model.llama_model.get_input_embeddings().weight.data, num_new_tokens)
            _noisy_mean_initialization(model.llama_model.get_output_embeddings().weight.data, num_new_tokens)

        logger.info("Resized token embeddings from {} to {}.".format(current_embedding_size, new_embedding_size))


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        losses, speech_output = model(**inputs)
        
        loss = losses.pop("loss")
        
        if losses and self.state.global_step % self.state.logging_steps ==0:
            self.log(losses)
        
        # return loss
        return (loss, speech_output) if return_outputs else loss


def main():
    # 1. Parse input arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train: # and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print("last_checkpoint", last_checkpoint)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 4. Load tokenizer
    if model_args.speech_llama:
        tokenizer = AutoTokenizer.from_pretrained(model_args.speech_llama, use_fast=True,)
        try:
            extractor = AutoFeatureExtractor.from_pretrained(model_args.speech_llama)
        except:
            extractor = AutoFeatureExtractor.from_pretrained(os.path.dirname(model_args.speech_llama))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.llama_model, use_fast=True,)
        extractor = AutoFeatureExtractor.from_pretrained(model_args.speech_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if "<|end-of-read|>" not in tokenizer.special_tokens_map or "<|end-of-write|>" not in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|end-of-read|>', '<|end-of-write|>', ]})

    # 5. Load dataset
    dataset = load_speech_to_text_paired_dataset(
        dataroot=data_args.data,
        manifest_files=data_args.manifest_files,
        tokenizer=tokenizer,
        instruction_prefix=data_args.instruction_prefix,
        instruction_suffix=data_args.instruction_suffix,
        speech_model_type=model_args.speech_model_type,
        task_type=data_args.task_type,
    )

    data_example = dataset[0]["input_ids"] + dataset[0]["suffix_input_ids"]
    print("=" * 80)
    print(f"input_ids:\n{data_example}")
    print(f'\nlabels:\n{dataset[0]["labels"] + dataset[0]["suffix_labels"]}')

    print(f"\n{tokenizer.decode(data_example)}")
    print("=" * 80)

    if data_args.eval_manifest_files is not None:
        eval_dataset = load_speech_to_text_paired_dataset(
            dataroot=data_args.data,
            manifest_files=data_args.eval_manifest_files,
            tokenizer=tokenizer,
            instruction_prefix=data_args.instruction_prefix,
            instruction_suffix=data_args.instruction_suffix,
            speech_model_type=model_args.speech_model_type,
            task_type=data_args.task_type,
        )
    else:
        eval_dataset = None

    # 6. Load pretrained model
    if model_args.speech_llama:
        model = SpeechLlamaModel.from_pretrained(
            model_args.speech_llama, 
            speech_label=model_args.speech_label,
            text_label=model_args.text_label,
            cfd_weight=model_args.cfd_weight,
        )
        model.init_decision_head()
        assert model.config.speech_model_type == model_args.speech_model_type
    else:
        SpeechEncoder = SpeechModels[model_args.speech_model_type]
        SpeechConfig = SpeechConfigs[model_args.speech_model_type]

        speech_config = SpeechConfig.from_pretrained(model_args.speech_model)
        llama_config = LlamaConfig.from_pretrained(model_args.llama_model)
        speech_llama_config = SpeechLlamaConfig(
            speech_config.to_dict(),
            llama_config.to_dict(),
            speech_model_type=model_args.speech_model_type,
            adapter_type=model_args.adapter_type,
            adapter_inner_dim=model_args.adapter_inner_dim,
            conv_kernel_sizes=model_args.conv_kernel_sizes,
            pad_id=tokenizer.pad_token_id,
            speech_label=model_args.speech_label,
            text_label=model_args.text_label,
            cfd_weight=model_args.cfd_weight,
        )

        model = SpeechLlamaModel(speech_llama_config)

        if model_args.speech_model_type == "wav2vec2":# or model_args.speech_model_type == "wav2vec_s" :
            dropout_configs = {
                "attention_dropout": 0.1,
                "activation_dropout": 0.0,
                "feat_extract_dropout": 0.0,
                "feat_proj_dropout": 0.1,
                "feat_quantizer_dropout": 0.1,
                "final_dropout": 0.0,
                "hidden_dropout": 0.0,
                "hidden_dropout_prob": 0.0,
                "layerdrop": 0.0,
                "apply_spec_augment": False
            }

            model.speech_model = SpeechEncoder.from_pretrained(model_args.speech_model, **dropout_configs)
        else:
            model.speech_model = SpeechEncoder.from_pretrained(model_args.speech_model)

        model.speech_config = model.speech_model.config
        model.llama_model = AutoModelForCausalLM.from_pretrained(
            model_args.llama_model, 
            attn_implementation="flash_attention_2",
            torch_dtype=llama_config.torch_dtype
        )

    resize_embedding_layer(model, tokenizer)
    
    if model.llama_config.vocab_size != len(tokenizer):
        model.llama_config.vocab_size = len(tokenizer)


    if is_main_process(training_args.local_rank):
        print("model.speech_config")
        print(model.speech_config)
        print("model.llama_config.config")
        print(model.llama_config)
        print(model.config)
        print(model)

    frozen_modules = model_args.frozen_modules.split(",")
    
    for frozen_module in frozen_modules:
        if frozen_module == "speech":
            for name, param in model.speech_model.named_parameters():
                param.requires_grad = False
        elif frozen_module == "llm":
            for name, param in model.llama_model.named_parameters():
                param.requires_grad = False
        elif frozen_module == "adapter":
            for name, param in model.adapter.named_parameters():
                param.requires_grad = False

    # 7. Define data collator
    data_collator = SpeechToTextPairedDataCollator(
        pad_id=tokenizer.pad_token_id,
        sampling_rate=extractor.sampling_rate,
        extractor=extractor,
        speech_model_type=model_args.speech_model_type,
    )

    # 8. Initialize Trainer
    # trainer = Trainer(
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer
    )

    # 9. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    tokenizer.save_pretrained(training_args.output_dir)
    extractor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()