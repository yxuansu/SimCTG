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

val_fct = CrossEntropyLoss(reduction='none')
class SimCTGGPT(nn.Module):
    def __init__(self, model_name, special_token_list=[]):
        super(SimCTGGPT, self).__init__()
        from transformers import AutoTokenizer, GPT2LMHeadModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
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

    def compute_logits_and_hidden_states(self, input_ids):
        # used for advanced decoding
        # input_ids: 1 x seqlen
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        return last_hidden_states, logits

    def forward(self, input_ids, labels):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.size() == torch.Size([bsz, seqlen, self.embed_dim])
        return last_hidden_states, logits

    def eval_loss(self, input_ids, labels):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        mle_loss = val_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        assert mle_loss.size() == torch.Size([bsz * seqlen])
        mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
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
        end_of_sequence_token_id = None, early_stop = False):
        '''
           input_ids: prefix input; 1 x prefix_len
           decoding_len: how many tokens to generate
           beam_width: size of candidate pool during decoding
           alpha: regulates importance of model confidence and degeneration penalty
           end_of_sequence_token_id: the token id that denotes the end of generation
           early_stop: whether to use the end_of_sequence_token_id to truncate the output
        '''
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        self.model.eval()
        from .utlisgpt import ContrastiveDecodingOneStepFast
        # sanity check
        assert alpha >= 0. and alpha <= 1.0
        
        # fast mode
        batch_size, seqlen = input_ids.size()
        prefix_len = seqlen
        #generated = [[] for _ in range(batch_size)]
        generated = [item for item in input_ids.tolist()]
        past_key_values = None
        last_hidden_states = None
        logits = None
        for step in range(decoding_len):
            input_ids, past_key_values, last_hidden_states, logits = ContrastiveDecodingOneStepFast(
                self.model,
                input_ids,
                beam_width,
                alpha,
                past_key_values,
                last_hidden_states,
                self.tokenizer,
                logits,
                first_step=step == 0,
            )
            tokens = input_ids.squeeze(dim=-1).tolist()
            for idx, t in enumerate(tokens):
                generated[idx].append(t)

        output = generated[0]
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if len(tmp) < prefix_len:
                    tmp.append(output[idx])
                else:
                    if output[idx] != end_of_sequence_token_id:
                        tmp.append(output[idx])
                    else:
                        break
            output = tmp
        return output

    def diverse_contrastive_search(self, input_ids, sample_step, nucleus_p, beam_width, alpha, decoding_len,
        end_of_sequence_token_id = None, early_stop = False):
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
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        contrastive_step = decoding_len - sample_step
        _, prefix_len = input_ids.size()
        # first do sample
        input_ids = self.model.generate(
                            input_ids, 
                            do_sample=True, 
                            max_length=prefix_len+sample_step, 
                            top_p=nucleus_p,
                            top_k=0)
        # then do contrastive search
        output = self.fast_contrastive_search(input_ids, beam_width, alpha, contrastive_step)
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if len(tmp) < prefix_len:
                    tmp.append(output[idx])
                else:
                    if output[idx] != end_of_sequence_token_id:
                        tmp.append(output[idx])
                    else:
                        break
            output = tmp
        return output

    def greedy_search(self, input_ids, decoding_len, end_of_sequence_token_id = None, early_stop = False):
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            max_length=prefix_len+decoding_len)
        output = output[0]
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if len(tmp) < prefix_len:
                    tmp.append(output[idx])
                else:
                    if output[idx] != end_of_sequence_token_id:
                        tmp.append(output[idx])
                    else:
                        break
            output = tmp
        return output

    def beam_search(self, input_ids, beam_width, decoding_len, end_of_sequence_token_id = None, early_stop = False):
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            max_length=prefix_len+decoding_len, 
                            num_beams=beam_width)
        output = output[0]
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if len(tmp) < prefix_len:
                    tmp.append(output[idx])
                else:
                    if output[idx] != end_of_sequence_token_id:
                        tmp.append(output[idx])
                    else:
                        break
            output = tmp
        return output

    def nucleus_sampling(self, input_ids, nucleus_p, decoding_len, end_of_sequence_token_id = None, early_stop = False):
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            do_sample=True, 
                            max_length=prefix_len+decoding_len, 
                            top_p=nucleus_p,
                            top_k=0)
        output = output[0]
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if len(tmp) < prefix_len:
                    tmp.append(output[idx])
                else:
                    if output[idx] != end_of_sequence_token_id:
                        tmp.append(output[idx])
                    else:
                        break
            output = tmp
        return output

    def topk_sampling(self, input_ids, topk, decoding_len, end_of_sequence_token_id = None, early_stop = False):
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            do_sample=True, 
                            max_length=prefix_len+decoding_len, 
                            top_p=1.0,
                            top_k=topk)
        output = output[0]
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if len(tmp) < prefix_len:
                    tmp.append(output[idx])
                else:
                    if output[idx] != end_of_sequence_token_id:
                        tmp.append(output[idx])
                    else:
                        break
            output = tmp
        return output
