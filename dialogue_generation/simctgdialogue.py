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
from loss_func import contrastive_loss

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

train_fct = CrossEntropyLoss()
val_fct = CrossEntropyLoss(reduction='none')
class SimCTGDialogue(nn.Module):
    def __init__(self, model_name, eos_token, pad_token):
        super(SimCTGDialogue, self).__init__()
        from transformers import AutoTokenizer, GPT2LMHeadModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.embed_dim = self.model.config.hidden_size
        if pad_token in self.tokenizer.vocab:
            print ('PAD token exists.')
        else:
            print ('Add PAD token to the tokenizer.')
            print ('Original vocabulary size is {}'.format(len(self.tokenizer)))
            self.tokenizer.add_tokens([pad_token])
            print ('Vocabulary size after extension is {}'.format(len(self.tokenizer)))
            assert len(self.tokenizer.convert_tokens_to_ids([pad_token])) == 1
            self.model.resize_token_embeddings(len(self.tokenizer)) 
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids([pad_token])[0]
        self.vocab_size = len(self.tokenizer)
        if 'e' in eos_token:
            self.eos_token = self.tokenizer.eos_token # English GPT
        else:
            self.eos_token = eos_token
        print (self.eos_token)
        #self.pad_token_id = pad_token_id

    def compute_logits_and_hidden_states(self, input_ids):
        # used for advanced decoding
        # input_ids: 1 x seqlen
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        return last_hidden_states, logits

    def forward(self, input_ids, labels, margin):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.size() == torch.Size([bsz, seqlen, self.embed_dim])
        mle_loss = train_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        norm_rep = last_hidden_states / last_hidden_states.norm(dim=2, keepdim=True)
        cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1,2)) 
        assert cosine_scores.size() == torch.Size([bsz, seqlen, seqlen])
        cl_loss = contrastive_loss(margin, cosine_scores, input_ids, self.pad_token_id, prefix_len=0)
        return mle_loss, cl_loss

    def eval_loss(self, input_ids, labels):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.size() == torch.Size([bsz, seqlen, self.embed_dim])
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

    # dialogue inference part
    def parse_dialogue_context(self, context_list, cuda_available=False, device=0):
        # context_list: a list of utterances in the dialogue session
        uttr_num = len(context_list)
        context_text = self.eos_token.join(context_list).strip(self.eos_token) + self.eos_token
        #print (context_text)
        tokens = self.tokenizer.tokenize(context_text)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = input_ids
        input_ids = torch.LongTensor(input_ids).view(1,-1)
        if cuda_available:
            input_ids = input_ids.cuda(device)
        return input_ids, uttr_num

    def extract_response(self, output_ids, uttr_num):
        output_text = self.tokenizer.decode(output_ids)
        # extract response
        item_list = output_text.split(self.eos_token)
        response = item_list[uttr_num].strip()
        if self.eos_token == '<|endoftext|>': # English GPT
            response = ' '.join(response.split())
        else:
            response = ''.join(response.split())
        return response

    def contrastive_search(self, context_list, beam_width, alpha, decoding_len, 
        cuda_available=False, device=0):
        input_ids, uttr_num = self.parse_dialogue_context(context_list, 
            cuda_available=cuda_available, device=device)
        output = self.fast_contrastive_generation(input_ids, beam_width, alpha, decoding_len)
        return self.extract_response(output, uttr_num)

    def diverse_contrastive_search(self, context_list, sample_step, nucleus_p, 
        beam_width, alpha, decoding_len, cuda_available=False, device=0):
        input_ids, uttr_num = self.parse_dialogue_context(context_list,
            cuda_available=cuda_available, device=device)
        output = self.diverse_contrastive_generation(input_ids, sample_step, nucleus_p, 
            beam_width, alpha, decoding_len)
        return self.extract_response(output, uttr_num)

    def greedy_search(self, context_list, decoding_len, cuda_available=False, device=0):
        input_ids, uttr_num = self.parse_dialogue_context(context_list,
            cuda_available=cuda_available, device=device)
        output = self.greedy_generation(input_ids, decoding_len)
        return self.extract_response(output, uttr_num)

    def beam_search(self, context_list, beam_width, decoding_len, 
        cuda_available=False, device=0):
        input_ids, uttr_num = self.parse_dialogue_context(context_list,
            cuda_available=cuda_available, device=device)
        output = self.beam_generation(input_ids, beam_width, decoding_len)
        return self.extract_response(output, uttr_num)

    def nucleus_sampling(self, context_list, nucleus_p, decoding_len, 
        cuda_available=False, device=0):
        input_ids, uttr_num = self.parse_dialogue_context(context_list,
            cuda_available=cuda_available, device=device)
        output = self.nucleus_generation(input_ids, nucleus_p, decoding_len)
        return self.extract_response(output, uttr_num)

    # decoding functions
    # ------------------------------------------------------- #
    def slow_contrastive_search(self, input_ids, beam_width, alpha, decoding_len):
        '''
           input_ids: prefix input; 1 x prefix_len
           decoding_len: how many tokens to generate
           beam_width: size of candidate pool during decoding
           alpha: regulates importance of model confidence and degeneration penalty
        '''
        # sanity check
        # sanity check
        assert alpha >= 0. and alpha <= 1.0

        from utlis import ContrastiveDecodingOneStep
        for step in range(decoding_len):
            input_ids = ContrastiveDecodingOneStep(self, input_ids, beam_width, alpha)
        return input_ids[0]

    def fast_contrastive_generation(self, input_ids, beam_width, alpha, decoding_len):
        '''
           input_ids: prefix input; 1 x prefix_len
           decoding_len: how many tokens to generate
           beam_width: size of candidate pool during decoding
           alpha: regulates importance of model confidence and degeneration penalty
        '''
        self.model.eval()
        from utlis import ContrastiveDecodingOneStepFast
        # sanity check
        assert alpha >= 0. and alpha <= 1.0
        
        # fast mode
        batch_size, seqlen = input_ids.size()
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
        return generated[0]

    def diverse_contrastive_generation(self, input_ids, sample_step, nucleus_p, beam_width, alpha, decoding_len):
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
        output = self.fast_contrastive_generation(input_ids, beam_width, alpha, contrastive_step)
        return output

    def greedy_generation(self, input_ids, decoding_len):
        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            max_length=prefix_len+decoding_len)
        return output[0]

    def beam_generation(self, input_ids, beam_width, decoding_len):
        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            max_length=prefix_len+decoding_len, 
                            num_beams=beam_width)
        return output[0]

    def nucleus_generation(self, input_ids, nucleus_p, decoding_len):
        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            do_sample=True, 
                            max_length=prefix_len+decoding_len, 
                            top_p=nucleus_p,
                            top_k=0)
        return output[0]
    # ------------------------------------------------------- #

    def compute_correlation_matrix(self, input_ids):        
        _, seq_len = input_ids.size()
        hidden = self.model.base_model(input_ids).last_hidden_state
        #print (hidden)
        norm_hidden = hidden / hidden.norm(dim=2, keepdim=True)
        correlation_matrix = torch.matmul(norm_hidden, norm_hidden.transpose(1,2)).view(seq_len, seq_len)
        return correlation_matrix.detach().numpy()

    # to produce similarity matrix heatmap
    def save_token_similarity_map(self, input_ids, save_name):
        input_ids = torch.LongTensor(input_ids).view(1, -1)
        correlation_matrix = self.compute_correlation_matrix(input_ids)
        df = pd.DataFrame(correlation_matrix)
        df.to_string(index=False)
        df.style.hide_index()
        df.style.hide_index()
        sns.heatmap(df, cmap="Blues", xticklabels=False, yticklabels=False)
        plt.savefig(save_name, format='png', dpi=500, bbox_inches = 'tight')
        plt.show()

