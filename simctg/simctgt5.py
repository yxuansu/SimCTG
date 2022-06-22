import sys
import os
import operator
from operator import itemgetter
import torch
from torch import nn
import random
import argparse
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ========== batch version ========= #
def ranking_fast(context_hidden, next_hidden, next_top_k_probs, alpha, beam_width):
    '''
        context_hidden: bsz*beam x seqlen x embed_dim
        next_hidden: bsz*beam x 1 x embed_dim
        next_top_k_probs: bsz x beam
    '''
    _, context_len, embed_dim = context_hidden.size()
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)    # [B*K, S]
    scores, _ = torch.max(cosine_matrix, dim=-1)    # [B*K]
    next_top_k_probs = next_top_k_probs.view(-1)    # [B*K]
    scores = (1.0 - alpha) * next_top_k_probs - alpha * scores 
    scores = torch.stack(torch.split(scores, beam_width))    # [B, K]
    selected_idx = scores.max(dim=-1)[1]    # [B]
    return selected_idx

def EncDecContrastiveDecodingOneStepFast(
    model, 
    ids, 
    decoder_ids,
    beam_width, 
    alpha, 
    past_key_values,
    last_hidden_states,
    vocab,
    logit_for_next_step,
    is_diverse,
    first_step=False,
    encoder_outputs=None,
    ):
    # input_ids: [B, S]
    if first_step:
        output = model(
            input_ids=ids, 
            decoder_input_ids=decoder_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=True
        )
        past_key_values = output.past_key_values
        last_hidden_states = output.decoder_hidden_states[-1]    # [B, S, E]
        logit_for_next_step = output.logits[:, -1, :]    # [B, V]
        # input_embeds = output.encoder_hidden_states[0]    # [B, S, E]
        # input_embeds = input_embeds.expand(beam_width, -1, -1)    # [5, S, E]
        encoder_outputs = (output.encoder_last_hidden_state, output.encoder_hidden_states, output.encoder_attentions)
    bsz, seqlen, embed_dim = last_hidden_states.size()

    next_probs = F.softmax(logit_for_next_step, dim=-1)
    _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width)    # [B, K]
    top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)    # [B, K]
    # compute new hidden
    past_key_values = encdec_enlarge_past_key_values(past_key_values, beam_width)
    output = model(
        # input_ids=torch.cat([ids for _ in range(beam_width)]),
        encoder_outputs=encoder_outputs,
        decoder_input_ids=top_k_ids.contiguous().view(-1, 1), 
        past_key_values=past_key_values,
        output_hidden_states=True,
        use_cache=True,
    )
    past_key_values = output.past_key_values
    logits = output.logits[:, -1, :]    # [B*K, V]
    next_hidden = output.decoder_hidden_states[-1]    # [B*K, 1, E]
    context_hidden = last_hidden_states.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(bsz*beam_width, seqlen, embed_dim)    # [B*K, S, E]

    if first_step:
        assert is_diverse in [True, False]
        if is_diverse:
            first_step_alpha = alpha
        else:
            first_step_alpha = 0.
        selected_idx = ranking_fast(
            context_hidden, 
            next_hidden, 
            top_k_probs,    # [B, K] 
            alpha=first_step_alpha, # if we are in the first decoding step, then we greedily select the highest prediction from the model by setting $\alpha$ as 0.0
            beam_width=beam_width,
        )     # [B]
    else:
        selected_idx = ranking_fast(
            context_hidden, 
            next_hidden, 
            top_k_probs,    # [B, K] 
            alpha,
            beam_width,
        )     # [B]
    # prepare for the next step
    next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)    # [B, 1]
    next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_width))    # [B, K, E]
    next_hidden = next_hidden[range(bsz), selected_idx, :]    # [B, E]
    last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)    # [B, S, E]
    past_key_values = encdec_select_past_key_values(past_key_values, beam_width, selected_idx)
    logits = torch.stack(torch.split(logits, beam_width))[range(bsz), selected_idx, :]    # [B, V]
    # next_id: [B, 1]
    return next_id, past_key_values, last_hidden_states, logits, encoder_outputs


def encdec_enlarge_past_key_values(past_key_values, beam_width):
    # from [B, num_head, seq_len, esz] to [B*K, num_head, seq_len, esz]
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            # item is the key and value matrix
            bsz, num_head, seq_len, esz = item.size()
            item = item.expand(beam_width, -1, -1, -1).contiguous()    # [bsz*beam, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(tuple(items))
    return tuple(new_key_values)

def encdec_select_past_key_values(past_key_values, beam_width, selected_idx):
    '''select_idx: [B]'''
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            bsz_and_beam, num_head, seq_len, esz = item.size()
            bsz = int(bsz_and_beam//beam_width)
            item = torch.stack(torch.split(item, beam_width, dim=0)).contiguous()    # [B, K, num_head, seq_len, esz] 
            item = item[range(bsz), selected_idx, :, :, :]   # [B, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values

val_fct = CrossEntropyLoss(reduction='none')
class SimCTGT5(nn.Module):
    def __init__(self, model_name, user_defined_model=None, user_defined_tokenizer=None, special_token_list=[]):
        super(SimCTGT5, self).__init__()
        '''
            user_defined_model: whether user would like to use self-defined model
            user_defined_tokenizer: whether user would like to use self-defined tokenizer

            if user_defined_model and user_defined_tokenizer 
        '''
        from transformers import AutoTokenizer, T5ForConditionalGeneration, T5TokenizerFast
        if user_defined_model is None:
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            print ('Use user defined model.')
            self.model = user_defined_model
        #self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if user_defined_tokenizer is None:
            self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        else:
            print ('Use user defined tokenizer.')
            self.tokenizer = user_defined_tokenizer
        if len(special_token_list) > 0:
            print ('Original vocabulary size is {}'.format(len(self.tokenizer)))
            print ('Adding special tokens...')
            self.tokenizer.add_tokens(special_token_list)
            print ('Special token added.')
            print ('Resizing language model embeddings...')
            self.model.resize_token_embeddings(len(self.tokenizer))
            print ('Language model embeddings resized.')
        self.vocab_size = len(self.tokenizer)
        print ('The vocabulary size of the language model is {}'.format(len(self.tokenizer)))
        self.embed_dim = self.model.config.hidden_size

    def forward(self, encoder_inputs, encoder_mask, decoder_inputs, decoder_labels):
        outputs = self.model(input_ids=encoder_inputs, attention_mask=encoder_mask, 
                              decoder_input_ids=decoder_inputs, labels=decoder_labels,
                             output_hidden_states=True)
        last_hidden_states = outputs.decoder_hidden_states[-1]
        logits = outputs.logits
        return last_hidden_states, logits

    def eval_loss(self, encoder_inputs, encoder_mask, decoder_inputs, decoder_labels):
        bsz, seqlen = decoder_inputs.size()
        outputs = self.model(input_ids=encoder_inputs, attention_mask=encoder_mask, 
                              decoder_input_ids=decoder_inputs, labels=decoder_labels,
                             output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        mle_loss = val_fct(logits.view(-1, self.vocab_size), decoder_labels.view(-1))

        assert mle_loss.size() == torch.Size([bsz * seqlen])
        mask_tmp = decoder_labels.masked_fill(~decoder_labels.eq(-100), 1.0)
        mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)
        # sum 
        mle_loss_sum = torch.sum(mle_loss)
        token_num_sum = torch.sum(mask)
        return mle_loss_sum, token_num_sum

    def save_model(self, ckpt_save_path):
        import os
        if os.path.exists(ckpt_save_path):
            pass
        else: # recursively construct directory
            os.makedirs(ckpt_save_path, exist_ok=True)
        # save model
        self.model.save_pretrained(ckpt_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(ckpt_save_path)

    # decoding functions
    # ------------------------------------------------------- #
    @torch.no_grad()
    def fast_contrastive_search(self, input_ids, beam_width, alpha, decoding_len,
        start_of_sequence_token_id = None, end_of_sequence_token_id = None, early_stop = True):
        '''
           input_ids: source input; 1 x prefix_len
           decoding_len: how many tokens to generate
           beam_width: size of candidate pool during decoding
           alpha: regulates importance of model confidence and degeneration penalty
        '''
        if end_of_sequence_token_id is None:
            end_of_sequence_token_id = self.tokenizer.eos_token_id

        #from .utlist5 import EncDecContrastiveDecodingOneStepFast
        # sanity check
        assert alpha >= 0. and alpha <= 1.0
        
        batch_size, seqlen = input_ids.size()
        generated = []
        past_key_values = None
        last_hidden_states = None
        logits = None
        input_embeds = None

        # start of sequence token
        if start_of_sequence_token_id is None:
            decoder_ids = torch.LongTensor([self.model.config.decoder_start_token_id]).unsqueeze(0)
        else:
            decoder_ids = torch.LongTensor([start_of_sequence_token_id]).unsqueeze(0)

        if input_ids.is_cuda:
            decoder_ids = decoder_ids.cuda(input_ids.get_device())

        for step in range(decoding_len):
            decoder_ids, past_key_values, last_hidden_states, logits, input_embeds = EncDecContrastiveDecodingOneStepFast(
                self.model,
                input_ids,
                decoder_ids,
                beam_width,
                alpha,
                past_key_values,
                last_hidden_states,
                self.tokenizer,
                logits,
                is_diverse=False,
                first_step=step == 0,
                encoder_outputs=input_embeds,
            )
            token = decoder_ids.squeeze(dim=-1).item()
            if early_stop:
                if token == end_of_sequence_token_id:
                    break
                else:
                    pass
            generated.append(token)

        output = generated
        return output

    def diverse_contrastive_search(self, input_ids, sample_step, nucleus_p, beam_width, alpha, decoding_len,
        start_of_sequence_token_id = None, end_of_sequence_token_id = None, early_stop = True):
        '''
            sample_step: 
                number of steps to decode with nucleus sampling, 
                for the remaining steps we use contrastive search
            decoding_len: 
                the total number of generated tokens
            beam_width: 
                size of candidate pool during decoding
            alpha: 
                regulates importance of model confidence and degeneration penalty

        '''
        if end_of_sequence_token_id is None:
            end_of_sequence_token_id = self.tokenizer.eos_token_id

        # first do sample
        sampled_output = self.nucleus_sampling(input_ids=input_ids, nucleus_p=nucleus_p, decoding_len=sample_step, 
            start_of_sequence_token_id = start_of_sequence_token_id, early_stop = False)

        decoder_ids = sampled_output.view(1,-1)
        if input_ids.is_cuda:
            decoder_ids = decoder_ids.cuda(input_ids.get_device())

        # then do contrastive search
        assert alpha >= 0. and alpha <= 1.0
        
        batch_size, seqlen = input_ids.size()
        generated = []
        past_key_values = None
        last_hidden_states = None
        logits = None
        input_embeds = None

        contrastive_step = decoding_len - sample_step
        for step in range(contrastive_step):
            decoder_ids, past_key_values, last_hidden_states, logits, input_embeds = EncDecContrastiveDecodingOneStepFast(
                self.model,
                input_ids,
                decoder_ids,
                beam_width,
                alpha,
                past_key_values,
                last_hidden_states,
                self.tokenizer,
                logits,
                is_diverse=True,
                first_step=step == 0,
                encoder_outputs=input_embeds,
            )
            token = decoder_ids.squeeze(dim=-1).item()
            if early_stop:
                if token == end_of_sequence_token_id:
                    break
                else:
                    pass
            generated.append(token)
        tmp_output = sampled_output.detach().cpu().tolist() + generated
        output = []
        for token in tmp_output:
            if token == self.model.config.decoder_start_token_id:
                pass
            else:
                output.append(token)
        return output

    def greedy_search(self, input_ids, decoding_len, 
        start_of_sequence_token_id = None, end_of_sequence_token_id = None, early_stop = True):
        if end_of_sequence_token_id is None:
            end_of_sequence_token_id = self.tokenizer.eos_token_id

        if start_of_sequence_token_id is None:
            output = self.model.generate(
                                input_ids=input_ids, 
                                max_length=decoding_len)
        else:
            output = self.model.generate(
                                input_ids=input_ids,
                                decoder_start_token_id = start_of_sequence_token_id, 
                                max_length=decoding_len)
        output = output[0]
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if output[idx] == self.model.config.decoder_start_token_id:
                    pass
                elif output[idx] == end_of_sequence_token_id:
                    break
                else:
                    tmp.append(output[idx])
            output = tmp
        return output

    def beam_search(self, input_ids, beam_width, decoding_len, 
        start_of_sequence_token_id = None, end_of_sequence_token_id = None, early_stop = True):
        if end_of_sequence_token_id is None:
            end_of_sequence_token_id = self.tokenizer.eos_token_id

        if start_of_sequence_token_id is None:
            output = self.model.generate(
                                input_ids, 
                                max_length=decoding_len, 
                                num_beams=beam_width)
        else:
            output = self.model.generate(
                                input_ids, 
                                decoder_start_token_id = start_of_sequence_token_id, 
                                max_length=decoding_len, 
                                num_beams=beam_width)

        output = output[0]
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if output[idx] == self.model.config.decoder_start_token_id:
                    pass
                elif output[idx] == end_of_sequence_token_id:
                    break
                else:
                    tmp.append(output[idx])
            output = tmp
        return output

    def nucleus_sampling(self, input_ids, nucleus_p, decoding_len, 
        start_of_sequence_token_id = None, end_of_sequence_token_id = None, early_stop = True):

        if end_of_sequence_token_id is None:
            end_of_sequence_token_id = self.tokenizer.eos_token_id


                # start of sequence token
        if start_of_sequence_token_id is None:
            output = self.model.generate(
                                input_ids=input_ids, 
                                do_sample=True, 
                                max_length=decoding_len,
                                top_p=nucleus_p,
                                top_k=0)
        else:
            output = self.model.generate(
                                input_ids=input_ids, 
                                decoder_start_token_id = start_of_sequence_token_id,
                                do_sample=True, 
                                max_length=decoding_len,
                                top_p=nucleus_p,
                                top_k=0)

        output = output[0]
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if output[idx] == self.model.config.decoder_start_token_id:
                    pass
                elif output[idx] == end_of_sequence_token_id:
                    break
                else:
                    tmp.append(output[idx])
            output = tmp
        return output
