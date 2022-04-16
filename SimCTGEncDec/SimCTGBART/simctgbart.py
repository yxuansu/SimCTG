import sys
import ipdb
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

train_fct = CrossEntropyLoss()
val_fct = CrossEntropyLoss(reduction='none')
class SimCTGBART(nn.Module):
    def __init__(self, model_name):
        super(SimCTGBART, self).__init__()
        from transformers import AutoTokenizer, BartForConditionalGeneration
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        self.embed_dim = self.model.config.hidden_size
        self.pad_token_id = self.tokenizer.pad_token_id
        
    @torch.no_grad()
    # decoding functions
    # ------------------------------------------------------- #
    def fast_contrastive_search(self, input_ids, decoder_ids, beam_width, alpha, decoding_len):
        '''
           input_ids: prefix input; 1 x prefix_len
           decoding_len: how many tokens to generate
           beam_width: size of candidate pool during decoding
           alpha: regulates importance of model confidence and degeneration penalty
        '''
        self.model.eval()
        from utlis import EncDecContrastiveDecodingOneStepFast
        # sanity check
        assert alpha >= 0. and alpha <= 1.0
        
        batch_size, seqlen = input_ids.size()
        generated = []
        past_key_values = None
        last_hidden_states = None
        logits = None
        input_embeds = None
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
                first_step=step == 0,
                input_embeds=input_embeds,
            )
            token = decoder_ids.squeeze(dim=-1).item()
            generated.append(token)
        return generated

    def greedy_search(self, input_ids, decoding_len):
        output = self.model.generate(
                            input_ids=input_ids, 
                            max_length=decoding_len)
        return output[0]

    def beam_search(self, input_ids, beam_width, decoding_len):
        output = self.model.generate(
                            input_ids=input_ids, 
                            max_length=decoding_len, 
                            num_beams=beam_width)
        return output[0]

