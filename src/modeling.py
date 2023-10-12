import random
import warnings
from typing import Dict, List, Optional, Tuple, Iterable
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
import copy
from .model_utils import GenerationMixinOld
from transformers.activations import ACT2FN
from transformers import BartConfig
from transformers import MBartConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
import collections
from sentence_transformers import SentenceTransformer, util
import time
from transformers import BartTokenizer, BartConfig
from transformers.models.bart.modeling_bart import *

# Copied and modified from https://github.com/neukg/KAT-TSLF
# we mainly update the code in BartForConditionalGeneration class
# with our proposed contrastive loss, SRKL loss & normal KL loss for ablation

class BartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig, args=None):
        super().__init__(config)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.encoder = BartEncoder(config, self.shared)
        self.gatew = nn.Linear(config.d_model * 2, 1)
        self.queryw = nn.Linear(config.d_model, config.d_model, bias=True)
        self.style_encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)
        self.init_weights()
        self.embed_dim = config.d_model
        
    def set_args(self, args):
        self.args = args
        self.decoder.set_args(args)

    # load weights for the style encoder, using same initial weight as content encoder
    def init_style_encoder_weights(self):
        encoder_dict = self.encoder.state_dict()
        style_encoder_dict = self.style_encoder.state_dict()
        for name in style_encoder_dict:
            style_encoder_dict[name].copy_(encoder_dict[name])

        decoder_dict = self.decoder.state_dict()
        for name in decoder_dict:
            if 'style_attn' in name:
                style_name = name.replace('style_attn', 'encoder_attn')
                decoder_dict[name].copy_(decoder_dict[style_name])
    
    def get_encoder_outputs(self, input_ids, attention_mask, 
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):

        content_outputs = self.encoder(
            input_ids=input_ids[0],
            attention_mask=attention_mask[0],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        content_hidden = content_outputs.last_hidden_state

        style_outputs = self.style_encoder(
            input_ids=input_ids[1],
            attention_mask=attention_mask[1],
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        initial_style_hidden = style_outputs.last_hidden_state
        # content_hidden: [B,Lc,d], initial_style_hidden: [B,Ls,d]

        # cross attention
        score_matrix = torch.bmm(content_hidden, initial_style_hidden.transpose(1,2))
        score_matrix = F.softmax(score_matrix, dim=-1) # [B,Lc,Ls]
        cross_attended = torch.bmm(score_matrix, initial_style_hidden) # [B,Lc,d]

        selected_style_hidden = cross_attended
        selected_attention_mask = torch.ones(selected_style_hidden.shape[:2]).to(selected_style_hidden.device)
        attention_mask = [attention_mask[0], selected_attention_mask] 
        style_outputs = BaseModelOutput(last_hidden_state = selected_style_hidden, hidden_states=None, attentions=None)

        return content_outputs, style_outputs, attention_mask, initial_style_hidden, selected_style_hidden, content_hidden
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        initial_style_hidden = None
        selected_style_hidden = None
        content_hidden = None

        if encoder_outputs is None:
            content_outputs, style_outputs, attention_mask, initial_style_hidden, selected_style_hidden, content_hidden = self.get_encoder_outputs(
                input_ids, attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            content_outputs, style_outputs = encoder_outputs
            

        decoder_outputs = self.decoder(
            decoder_input_ids,
            [content_outputs['last_hidden_state'], style_outputs['last_hidden_state']],
            [attention_mask[0], attention_mask[1]],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                encoder_last_hidden_state=[content_outputs.last_hidden_state, style_outputs.last_hidden_state],
                encoder_hidden_states=[content_outputs.hidden_states, style_outputs.hidden_states],
                encoder_attentions=[content_outputs.attentions, style_outputs.attentions],
        ), initial_style_hidden, selected_style_hidden, attention_mask, content_hidden

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)  # make it on the fly

class BartForConditionalGeneration(PretrainedBartModel, GenerationMixinOld):
    base_model_prefix = "model"
    authorized_missing_keys = [r"final_logits_bias", r"encoder\.version", r"decoder\.version"]

    def __init__(self, config: BartConfig, args):
        super().__init__(config)
        base_model = BartModel(config, args)
        self.model = base_model
        self.config = config
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

        # visualization
        self.r = []
        self.us = []
        self.s = []
    
    def set_args(self, args):
        self.args = args
        self.model.set_args(args)

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        old_num_tokens = self.model.shared.num_embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.model.shared = new_embeddings

        self.model.encoder.embed_tokens = new_embeddings
        self.model.style_encoder.embed_tokens = new_embeddings
        
        self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int) -> None:
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

  
    def modify_inp_and_tgt(self, tgt_ids, pad_token_id, eos_token_id):
        prev_output_tokens = tgt_ids.clone()
        index_of_eos = (tgt_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
        prev_output_tokens[:, 0] = tgt_ids.gather(1, index_of_eos).squeeze()
        prev_output_tokens[:, 1:] = tgt_ids[:, :-1]
       
        return prev_output_tokens, tgt_ids

    def modify_tgt_mask(self, decoder_attention_mask):
        # return decoder_attention_mask[:, :-1]
        return decoder_attention_mask

    def prepare_input(
        self,
        decoder_input_ids,
        past=None,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return decoder_input_ids
    
    def get_mean(self, x):
        return torch.mean(x, dim = 1, keepdim = False)
    
    def kl_loss(self, response_hidden, context_hidden, initial_style_hidden, attention_mask):
        kl_loss = nn.KLDivLoss(reduction="none", log_target=True)
        softmax = torch.nn.Softmax(dim=-1) 
        def KL_div(x,y):
            return kl_loss(x.log(),y.log())

        response_hidden = self.get_mean(response_hidden) #[B,d]
        context_hidden = self.get_mean(context_hidden) #[B,d]
        initial_style_hidden = initial_style_hidden #[B,L,d]
        context_distribution = softmax(torch.bmm(context_hidden.unsqueeze(1), initial_style_hidden.transpose(1,2)).squeeze(1)) #[B,L]
        response_distribution = softmax(torch.bmm(response_hidden.unsqueeze(1), initial_style_hidden.transpose(1,2)).squeeze(1)) #[B,L]

        loss = KL_div(context_distribution, response_distribution)
        return loss.mean()
    
    def contrastive_loss(self, response_hidden, initial_style_hidden, selected_style_hidden):
        def get_mean_normalized(x):
            x = torch.mean(x, dim = 1, keepdim = False)
            return x / torch.norm(x, dim = -1, keepdim = True)      

        loss = 0
        loss_fct = nn.CrossEntropyLoss(reduction='mean')

        mean_response_hidden = get_mean_normalized(response_hidden) #[B,d]
        mean_initial_style_hidden = get_mean_normalized(initial_style_hidden) #[B,d]
        mean_selected_style_hidden = get_mean_normalized(selected_style_hidden) #[B,d]

        mean_style_hidden = torch.cat([mean_selected_style_hidden, mean_initial_style_hidden], dim=0) #[2B,d]
        score_map = torch.matmul(mean_response_hidden, mean_style_hidden.transpose(0,1)) #[B,2B]
        score_map = score_map / self.args.temperature
        loss = loss_fct(score_map, torch.arange(score_map.shape[0]).to(score_map.device))
        
        return loss

    def SRKL_loss(self, response_hidden, context_hidden, selected_style_hidden):
        mean_response_hidden = self.get_mean(response_hidden) # [B,d]
        mean_context_hidden = self.get_mean(context_hidden) # [B,d]
        mean_selected = self.get_mean(selected_style_hidden) # [B,d]

        # direction loss
        style_direction = mean_context_hidden-mean_response_hidden
        dir_loss = torch.bmm(mean_selected.unsqueeze(1),style_direction.unsqueeze(2)).squeeze(2)
        
        # style responsiveness loss
        resp_c = torch.bmm(selected_style_hidden, mean_context_hidden.unsqueeze(2)).squeeze(2)
        resp_c = torch.max(resp_c,dim=-1)[0] # [B]
        resp_r = torch.bmm(selected_style_hidden, mean_response_hidden.unsqueeze(2)).squeeze(2)
        resp_r = torch.max(resp_r,dim=-1)[0] # [B]
        resp_loss = resp_c - resp_r

        # reject < 0 loss
        dir_loss[dir_loss<0.0] = 0.0
        resp_loss[resp_loss<0.0] = 0.0

        SRKL_loss = self.args.lambda_dir * dir_loss + self.args.lambda_resp * resp_loss
        SRKL_loss = SRKL_loss.mean()

        return dir_loss, resp_loss, SRKL_loss

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        kp_mask=None,
        **unused,
    ):
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        decoder_attention_mask = None
        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        outputs, initial_style_hidden, selected_style_hidden, attention_mask, context_hidden = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )
        # logger.info(f"{outputs[0].shape}")
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        response_hidden = outputs[0]

        dialogue_loss = None
        contrastive_loss = None
        total_loss = 0
        return_loss = {}

        if labels is not None:
            loss_fct = CrossEntropyLoss()

            dialogue_loss = loss_fct(lm_logits.reshape(-1, self.config.vocab_size), labels.reshape(-1))
            contrastive_loss = self.contrastive_loss(outputs[0], initial_style_hidden, selected_style_hidden)
            if self.args.use_KL:
                kl_loss = self.kl_loss(response_hidden, context_hidden, initial_style_hidden, attention_mask)
            dir_loss, resp_loss, SRKL_loss = self.SRKL_loss(response_hidden, context_hidden, selected_style_hidden)
            
            return_loss['dialogue_loss'] = dialogue_loss
            return_loss['contrastive_loss'] = contrastive_loss
            return_loss['SRKL_loss'] = SRKL_loss
            return_loss['dir_loss'] = dir_loss.mean()
            return_loss['resp_loss'] = resp_loss.mean()

            if self.args.use_KL:
                total_loss += kl_loss
                return_loss['kl_loss'] = kl_loss
            
            total_loss += dialogue_loss + self.args.lambda_contrastive * contrastive_loss + SRKL_loss

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((return_loss,) + output) if dialogue_loss is not None else output

        return_loss['total_loss'] = total_loss

        return Seq2SeqLMOutput(
            loss=return_loss if dialogue_loss is not None else None,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs
    ):
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1:
            self._force_token_ids_generation(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(self.config.vocab_size) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = []
        for layer_past in past:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)
        return reordered_past

    def get_encoder(self):
        return self.model.encoder, self.model.style_encoder

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.model.shared)  # make it on the fly

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_kwargs
    ) -> torch.LongTensor:
        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            batch_size = input_ids[0].shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
            bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        assert input_ids is not None
        # assert len(input_ids) == 2
        # assert input_ids[0].dim() == 2
        # assert input_ids[1].dim() == 3
        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids[0]):
            attention_mask = [input_ids[0].ne(pad_token_id).long(), input_ids[1].ne(pad_token_id).long()]
        elif attention_mask is None:
            attention_mask = [input_ids[0].new_ones(input_ids[0].shape), input_ids[1].new_ones(input_ids[1].shape)]
        # if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
        #     attention_mask = input_ids.ne(pad_token_id).long()
        # elif attention_mask is None:
        #     attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            # logger.warning(
            #     "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
            # )
            pad_token_id = eos_token_id

        # current position and vocab size
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (
            self.config.is_encoder_decoder
            and hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "vocab_size")
        ):
            vocab_size = self.config.decoder.vocab_size

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                # see if BOS token can be used for decoder_start_token_id
                if bos_token_id is not None:
                    decoder_start_token_id = bos_token_id
                elif hasattr(self.config, "decoder") and hasattr(self.config.decoder, "bos_token_id"):
                    decoder_start_token_id = self.config.decoder.bos_token_id
                else:
                    raise ValueError(
                        "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
                    )

            assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

            # get encoder and store encoder outputs

            content_outputs, style_outputs, attention_mask, _,_,_ = self.model.get_encoder_outputs(
                input_ids, attention_mask,
                return_dict=True,
            )
            encoder_outputs = [content_outputs, style_outputs]

            # encoder_outputs: ModelOutput = encoder(input_ids[0], attention_mask=attention_mask[0], return_dict=True)

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams >= 1:
            src_attn_mask, style_attn_mask = attention_mask 
            src_ids_len = src_attn_mask.shape[-1]
            style_ids_len = style_attn_mask.shape[-1]

            src_attn_mask = src_attn_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, src_ids_len
            )
            style_attn_mask = style_attn_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, style_ids_len
            )

            src_attn_mask = src_attn_mask.contiguous().view(
                effective_batch_size * num_beams, src_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            style_attn_mask = style_attn_mask.contiguous().view(
                effective_batch_size * num_beams, style_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams * num_style, cur_len)
            attention_mask = (src_attn_mask, style_attn_mask)  
            # print("generate:", attention_mask[0].shape, attention_mask[1].shape)
            
        if self.config.is_encoder_decoder:
            # create empty decoder_input_ids
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            cur_len = 1

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids[0].device)
            )
            # print(expanded_batch_idxs.shape)
            # print(encoder_outputs[0].last_hidden_state.shape, encoder_outputs[1].last_hidden_state.shape)

            hidden_size = encoder_outputs[1].last_hidden_state.shape[-1]
            tmp = encoder_outputs[1].last_hidden_state.view(batch_size, style_ids_len, hidden_size)
            encoder_outputs[0]["last_hidden_state"] = encoder_outputs[0].last_hidden_state.index_select(
                0, expanded_batch_idxs
            )
            encoder_outputs[1]["last_hidden_state"] = tmp.index_select(
                0, expanded_batch_idxs
            )

            # print(encoder_outputs[0]["last_hidden_state"].shape, encoder_outputs[1]["last_hidden_state"].shape)
            # assert 0
            # save encoder_outputs in `model_kwargs`
            model_kwargs["encoder_outputs"] = encoder_outputs

        else:
            cur_len = input_ids.shape[-1]

        assert (
            cur_len < max_length
        ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_kwargs=model_kwargs,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_kwargs=model_kwargs,
            )

        return output


# Unmodified part

tlen_for_abl = 0

_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"


BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-base",
    "facebook/bart-large",
    "facebook/bart-large-mnli",
    "facebook/bart-large-cnn",
    "facebook/bart-large-xsum",
    "facebook/mbart-large-en-ro",
]

def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    assert attention_mask.dim() == 2, f"Attn mask: {attention_mask.shape}"
    return attention_mask.eq(0)

def _prepare_bart_decoder_inputs(
    config, input_ids, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=torch.float32
):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id
    decoder_start_token_id = config.decoder_start_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id)
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    if decoder_padding_mask is not None and decoder_padding_mask.shape[1] > 1:
        # never mask leading token, even if it is pad
        decoder_padding_mask[:, 0] = decoder_padding_mask[:, 1]
    causal_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
        dtype=causal_mask_dtype, device=decoder_input_ids.device
    )
    return decoder_input_ids, decoder_padding_mask, causal_mask


class PretrainedBartModel(PreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs

def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer

# Helper Functions, mostly for making masks
def _check_shapes(shape_1, shape2):
    if shape_1 != shape2:
        raise AssertionError("shape mismatch: {} != {}".format(shape_1, shape2))

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask

class EncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Attention(self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout)
        self.normalize_before = config.normalize_before
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask, output_attentions=False):
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, attn_weights = self.self_attn(
            query=x, key=x, key_padding_mask=encoder_padding_mask, output_attentions=output_attentions
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn_weights

class BartEncoder(nn.Module):

    def __init__(self, config: BartConfig, embed_tokens):
        super().__init__()

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings,
                embed_dim,
                self.padding_idx,
                2,
        )
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.normalize_before else None

    def forward(
        self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False
    ):
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = [] if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)

            if output_attentions:
                all_attentions = all_attentions + (attn,)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)
            # T x B x C -> B x T x C
            encoder_states = tuple(hidden_state.transpose(0, 1) for hidden_state in encoder_states)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions)

class DecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = Attention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = Attention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention='src_decoder',
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        # modify
        self.style_attn = Attention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention='style_decoder', 
        )
        self.style_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        # self.style_gate = nn.Parameter(torch.randn(1))

        self.style_gate_w = nn.Linear(self.embed_dim * 2, 1)


    def forward(
        self,
        x,
        encoder_hidden_states,
        lid,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        decoder_padding_mask=None,
        output_attentions=False,
        use_style_attention=1,
    ):
        src_hidden_states, style_hidden_states = encoder_hidden_states
        src_hidden_attn_mask, style_attn_mask = encoder_attn_mask

        residual = x

        if layer_state is None:
            layer_state = {}

        # Self Attention
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            output_attentions=output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x,
            key=src_hidden_states,
            key_padding_mask=src_hidden_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)


        residual = x
        style_attn_weight=None
        if use_style_attention:
            # Style attention

            assert self.encoder_attn.cache_key != self.self_attn.cache_key
            if self.normalize_before:
                x = self.style_attn_layer_norm(x)

            x, style_attn_weight = self.style_attn(
                query=x,
                key=style_hidden_states,
                key_padding_mask=style_attn_mask,
                layer_state=layer_state,  # mutates layer state
                output_attentions=True
            )

            # Controller
            x = F.dropout(x, p=self.dropout, training=self.training)
            weight = self.style_gate_w(torch.cat([x, residual], dim=-1)).sigmoid()
            x = (1 - weight) * residual + weight * x
            # x = residual + x

            if not self.normalize_before:
                x = self.style_attn_layer_norm(x)
            
            x = x + residual
        else:
            x = residual

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            style_attn_weight,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding

class BartDecoder(nn.Module):
    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        self.embed_positions = LearnedPositionalEmbedding(
                    config.max_position_embeddings,
                    config.d_model,
                    self.padding_idx,
                    2,
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None
    
    def set_args(self,args):
        self.args=args

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        **unused,
    ):
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")

        # check attention mask and invert
        for i in range(len(encoder_padding_mask)):
            if encoder_padding_mask[i] is not None:
                # print("decoder phase", encoder_padding_mask[i].shape)
                encoder_padding_mask[i] = invert_mask(encoder_padding_mask[i])
        # embed positions
        # logger.info(input_ids.shape)
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()
        # logger.info(input_ids.shape)
        x = self.embed_tokens(input_ids) * self.embed_scale
        # print(x.shape, positions.shape)
        x += positions
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = [ehc.transpose(0, 1) for ehc in encoder_hidden_states]
        # encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        # logger.info(x.shape)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = []
        # ddddd = []
        # weight_for_abl.append([])
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_state = past_key_values[idx] if past_key_values is not None else None
            # logger.info(x.shape)
            x, layer_self_attn, layer_style_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                idx + 1,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
                output_attentions=output_attentions,
                # use_style_attention=self.args.use_style_attention,
            )
            # print(layer_kno_attn.shape)
            # ddddd.append(layer_kno_attn.reshape(1, 12, 1, 5, 16).sum(dim=-1).sum(dim=1).squeeze(1).softmax(dim=-1))
            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # if config.add_final_layer_norm (mBART)
                x = self.layer_norm(x)
            if output_attentions:
                all_self_attns += (layer_self_attn,)
       
        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        if output_hidden_states:
            all_hidden_states = tuple(hidden_state.transpose(0, 1) for hidden_state in all_hidden_states)
        x = x.transpose(0, 1)
        encoder_hidden_states = [ehc.transpose(0, 1) for ehc in encoder_hidden_states]
        # encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [x, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=x, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns
        )


def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache

class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention='self',  # otherwise self_attention
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = self.encoder_decoder_attention
        # self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"


    def _shape(self, tensor, seq_len, bsz):
        return tensor.contiguous().view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
        output_attentions=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv: bool = 'self' not in self.encoder_decoder_attention

        # static_kv: bool = self.encoder_decoder_attention != 'self'
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key" in saved_state and static_kv:
                # previous time steps are cached - no need to recompute key and value if they are static
                key = None
        else:
            saved_state = None
            layer_state = {}

        q = self.q_proj(query) * self.scaling
        if static_kv:
            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)

        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)

        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, static_kv, bsz)

        # Update cache
        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask if not static_kv else None,
        }

        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        # if key_padding_mask is not None:
        #     print(key_padding_mask.size(), bsz, src_len, k.shape, v.shape, self.cache_key)

        assert key_padding_mask is None or key_padding_mask.size()[:2] == (
            bsz,
            src_len,
        )

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.masked_fill(reshaped, -10000.0)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(
            attn_weights,
            p=self.dropout,
            training=self.training,
        )
        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if output_attentions:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None
        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get("prev_key_padding_mask", None)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = key_padding_mask
        return k, v, new_key_padding_mask

class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, offset):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = offset
        assert padding_idx is not None
        num_embeddings += offset
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(seq_len - 1)  # called before slicing
        else:
            # starts at 0, ends at 1-seq_len
            positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions + self.offset)

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

def _get_shape(t):
    return getattr(t, "shape", None)