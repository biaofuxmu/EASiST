import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from transformers import PreTrainedModel, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaConfig

try:
    from .configuration_speech_llama import SpeechLlamaConfig
    from .modeling_speech_model import SpeechModels, SpeechConfigs
    from .modeling_adapter import build_adapter
except:
    from configuration_speech_llama import SpeechLlamaConfig
    from modeling_speech_model import SpeechModels, SpeechConfigs
    from modeling_adapter import build_adapter

# import pdb;pdb.set_trace()

def convert_indices_to_sizes(split_indices, total_length):
    split_indices = sorted(split_indices)
    split_indices = [0] + split_indices
    split_sizes = [
        split_indices[i+1] - split_indices[i] if i < len(split_indices) - 2
        else total_length - split_indices[i]
        for i in range(len(split_indices) - 1)
    ]
    
    return split_sizes


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


class SpeechLlamaModel(PreTrainedModel):
    config_class = SpeechLlamaConfig
    base_model_prefix = "speech_llama"

    def __init__(self, config: SpeechLlamaConfig):
        super().__init__(config)

        self.config = config

        self.speech_config = SpeechConfigs[config.speech_model_type](**config.speech_config)
        self.llama_config = LlamaConfig(**config.llama_config)

        self.speech_model = SpeechModels[config.speech_model_type](self.speech_config)

        self.adapter = build_adapter(config, self.llama_config, self.speech_config)
        self.llama_model = AutoModelForCausalLM.from_config(self.llama_config)

        self.pad_id = config.pad_id
        self.speech_label = config.speech_label
        self.text_label = config.text_label
        self.cfd_weight = config.cfd_weight

        if self.speech_label:
            self.decision_head = nn.Linear(self.llama_config.hidden_size, 2, bias=False)

    def init_decision_head(self):
        if self.speech_label:
            nn.init.kaiming_uniform_(self.decision_head.weight, a=math.sqrt(5))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        speech_inputs: Optional[torch.FloatTensor] = None,
        suffix_input_ids: Optional[torch.LongTensor] = None,
        suffix_attention_mask: Optional[torch.LongTensor] = None,
        suffix_labels: Optional[torch.LongTensor] = None,
        chunk_seg_time: Optional[torch.LongTensor] = None,
        text_seg_sizes: Optional[torch.LongTensor] = None,
        is_streamings: Optional[List[bool]] = None,
    ):
        if chunk_seg_time is None:
            ### offline task
            ### 1. forward speech
            speech_embeds, speech_attention_mask = self.get_speech_features(speech_inputs)
            speech_labels = torch.LongTensor(speech_embeds.size(0), speech_embeds.size(1)).fill_(-100).to(speech_embeds.device)

            ### 2. forward llama
            prefix_embeds = self.llama_model.get_input_embeddings()(input_ids)
            suffix_embeds = self.llama_model.get_input_embeddings()(suffix_input_ids)
            
            inputs_embeds = torch.cat([prefix_embeds, speech_embeds, suffix_embeds], dim=1)
            attention_mask = torch.cat([attention_mask, speech_attention_mask, suffix_attention_mask], dim=1)
            labels = torch.cat([labels, speech_labels, suffix_labels], dim=1)
            
            decision_labels = None

        else:
            ### streaming task
            ### 1. forward speech
            speech_embeds, speech_attention_mask = self.get_speech_features(speech_inputs)
            speech_labels = torch.LongTensor(speech_embeds.size(0), speech_embeds.size(1)).fill_(-100).to(speech_embeds.device)

            ### 2. forward llama
            prompt_embeds = self.llama_model.get_input_embeddings()(input_ids)
            text_embeds = self.llama_model.get_input_embeddings()(suffix_input_ids)

            inputs_embeds, attention_mask, labels, decision_labels = self.interleave_triple_embeddings(
                prompt_embeds=prompt_embeds, 
                speech_embeds=speech_embeds, 
                text_embeds=text_embeds, 
                prompt_attn_mask=attention_mask, 
                speech_attn_mask=speech_attention_mask, 
                text_attn_mask=suffix_attention_mask, 
                prompt_labels=labels, 
                speech_labels=speech_labels, 
                text_labels=suffix_labels, 
                chunk_seg_time=chunk_seg_time,
                text_seg_sizes=text_seg_sizes,
                is_streamings=is_streamings
            )

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            output_hidden_states=True,
        )

        if decision_labels is not None:
            decision_logits = self.decision_head(outputs.hidden_states[-1])
            shift_logits = decision_logits[..., :-1, :].contiguous()
            shift_labels = decision_labels[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)

            decision_losses = nn.CrossEntropyLoss(reduction='none')(shift_logits, shift_labels)

            pad_mask = (shift_labels != -100).float()
            sum_loss = (decision_losses * pad_mask).sum()
            tokens = pad_mask.sum()

            decision_loss = sum_loss / (tokens + 1e-8)

            losses = {}
            losses["loss"] = outputs.loss + decision_loss * self.cfd_weight
            losses["total_loss"] = losses["loss"].item()
            losses["ce_loss"] = outputs.loss.item()
            losses["decision_loss"] = decision_loss.item()

            return losses, outputs
        else:
            return {"loss": outputs.loss}, outputs 

    def interleave_triple_embeddings(
        self,
        prompt_embeds: torch.LongTensor = None,
        speech_embeds: torch.LongTensor = None,
        text_embeds: torch.LongTensor = None,
        prompt_attn_mask: torch.LongTensor = None,
        speech_attn_mask: torch.LongTensor = None,
        text_attn_mask: torch.LongTensor = None,
        prompt_labels: torch.LongTensor = None,
        speech_labels: torch.LongTensor = None,
        text_labels: torch.LongTensor = None,
        chunk_seg_time: torch.LongTensor = None,
        text_seg_sizes: torch.LongTensor = None,
        is_streamings: Optional[List[bool]] = None,
    ):
        """
        prompt_embeds: bsz * prompt_len * dim
        speech_embeds: bsz * speech_len * dim
        text_embeds: bsz * text_len * dim
        chunk_seg_time: bsz * chunk_num
        """
        bsz = speech_embeds.shape[0]
        dim = speech_embeds.shape[2]
        shapes = [prompt_embeds.shape[0], speech_embeds.shape[0], text_embeds.shape[0], chunk_seg_time.shape[0]]
        assert all(shape == shapes[0] for shape in shapes), f"The first dimensions of all tensors should be equal, but got {shapes}"

        pad_emb = self.llama_model.get_input_embeddings()(torch.tensor(self.pad_id, device=text_embeds.device))

        interleaved_embeds = []
        interleaved_attn_masks = []
        interleaved_labels = []

        if self.speech_label:
            interleaved_decisions = []

        block_size = int(self.speech_model.encoder.main_context / (len(self.config.conv_kernel_sizes) * 2))

        for i in range(bsz):
            speech_embed = speech_embeds[i]
            s_attn_mask = speech_attn_mask[i]
            speech_label = speech_labels[i]
            seg_time = [idx for idx in chunk_seg_time[i] if idx !=-1]
            
            prompt_embed = prompt_embeds[i]
            prompt_label = prompt_labels[i]
            p_attn_mask = prompt_attn_mask[i]

            text_embed = text_embeds[i]
            t_attn_mask = text_attn_mask[i]
            text_label = text_labels[i]
            text_seg_size = [size for size in text_seg_sizes[i] if size != -1]

            prompt_len = p_attn_mask.sum().item()
            speech_len = s_attn_mask.sum().item()
            text_len = t_attn_mask.sum().item()

            is_streaming = is_streamings[i]

            if is_streaming:
                enc_seg_lens = [self.speech_model._get_feat_extract_output_lengths(input_lengths=idx, add_adapter=False) for idx in seg_time]
                adapter_seg_lens = [self.adapter.get_out_seq_lens_tensor(seg_len) for seg_len in enc_seg_lens]
                chunk_seg = convert_indices_to_sizes(adapter_seg_lens, speech_len)
                speech_seg_embs = torch.split(speech_embed[:speech_len], chunk_seg, dim=0)

                text_seg_embs = torch.split(text_embed[:text_len], text_seg_size, dim=0)
                text_seg_labels = torch.split(text_label[:text_len], text_seg_size, dim=0)

                response_embed = [torch.cat((s_emb, t_emb), dim=0) for s_emb, t_emb in zip(speech_seg_embs, text_seg_embs)]
                response_label = [torch.tensor(s_size * [-100] + text_seg_label.tolist(), device=text_seg_label.device) for s_size, text_seg_label in zip(chunk_seg, text_seg_labels)]
                
                interleaved_embed = torch.cat([prompt_embed[:prompt_len]] + response_embed, dim=0)
                interleaved_label = torch.cat([prompt_label[:prompt_len]] + response_label, dim=0)
                interleaved_attn_mask = torch.tensor([1] * interleaved_label.shape[0], device=text_embeds.device)

                if self.speech_label:
                    if self.text_label:
                        response_decision = [torch.tensor([0 if i % block_size == 0 and i > 0 else -100 for i in range(s_size)] + [1] * len(text_seg_label), device=text_seg_label.device) for s_size, text_seg_label in zip(chunk_seg, text_seg_labels)]
                    else:
                        response_decision = [torch.tensor([0 if i % block_size == 0 and i > 0 else -100 for i in range(s_size)] + [1] + [-100] * (len(text_seg_label) - 1), device=text_seg_label.device) for s_size, text_seg_label in zip(chunk_seg, text_seg_labels)]
                    decisions = torch.cat([prompt_label[:prompt_len]] + response_decision, dim=0)
                    interleaved_decisions.append(decisions)
            else:
                interleaved_embed = torch.cat([prompt_embed[:prompt_len], speech_embed[:speech_len], text_embed[:text_len]], dim=0)
                interleaved_label = torch.cat([prompt_label[:prompt_len], speech_label[:speech_len], text_label[:text_len]], dim=0)
                interleaved_attn_mask = torch.tensor([1] * interleaved_label.shape[0], device=text_embeds.device)
                
                if self.speech_label:
                    decisions = torch.zeros_like(interleaved_label).fill_(-100)
                    interleaved_decisions.append(decisions)

            interleaved_embeds.append(interleaved_embed)
            interleaved_labels.append(interleaved_label)
            interleaved_attn_masks.append(interleaved_attn_mask)

        max_len = max(emb.size(0) for emb in interleaved_embeds)

        inputs_embeds = pad_emb.repeat(bsz, max_len, 1)
        attention_mask = torch.zeros((bsz, max_len), dtype=text_attn_mask.dtype, device=speech_embeds.device)
        labels = torch.zeros((bsz, max_len), dtype=text_labels.dtype, device=speech_embeds.device).fill_(-100)

        for i, (emb, attn_mask, label) in enumerate(zip(interleaved_embeds, interleaved_attn_masks, interleaved_labels)):
            inputs_embeds[i, :emb.size(0)] = emb
            attention_mask[i, :attn_mask.size(0)] = attn_mask
            labels[i, :label.size(0)] = label

        assert labels.shape[1] == inputs_embeds.shape[1] and inputs_embeds.shape[0] == attention_mask.shape[0]

        if self.speech_label:
            decision_labels = torch.zeros((bsz, max_len), dtype=text_labels.dtype, device=speech_embeds.device).fill_(-100)
            for i, decision in enumerate(interleaved_decisions):
                decision_labels[i, :decision.size(0)] = decision
        else:
            decision_labels = None

        return inputs_embeds, attention_mask, labels, decision_labels


    def get_speech_features(self, speech_inputs):
        output = self.speech_model(**speech_inputs)
        speech_embeds = output.last_hidden_state # B x T x C
    
        speech_lengths = self.speech_model._get_feat_extract_output_lengths(speech_inputs.attention_mask.sum(-1))

        speech_embeds, speech_padding_mask = self.adapter(speech_embeds, speech_lengths)

        speech_attention_mask = ~speech_padding_mask

        return speech_embeds, speech_attention_mask

    @torch.no_grad()
    def generate(
        self,
        prefix_inputs,
        suffix_inputs,
        speech_inputs,
        generation_config={}
    ):
        inputs_embeds, attention_mask = [], []

        input_ids = prefix_inputs.input_ids.cuda()
        prefix_attention_mask = prefix_inputs.attention_mask.cuda()

        prefix_embeds = self.llama_model.get_input_embeddings()(input_ids)

        inputs_embeds.append(prefix_embeds)
        attention_mask.append(prefix_attention_mask)

        speech_embeds, speech_attention_mask = self.get_speech_features(speech_inputs)
        inputs_embeds.append(speech_embeds)
        attention_mask.append(speech_attention_mask)

        suffix_input_ids = suffix_inputs.input_ids.cuda()
        suffix_attention_mask = suffix_inputs.attention_mask.cuda()

        suffix_embeds = self.llama_model.get_input_embeddings()(suffix_input_ids)

        inputs_embeds.append(suffix_embeds)
        attention_mask.append(suffix_attention_mask)

        inputs_embeds = torch.cat(inputs_embeds, dim=1)
        attention_mask = torch.cat(attention_mask, dim=1)

        return self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_config
        )
