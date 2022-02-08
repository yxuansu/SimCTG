import sys
import os
import operator
from operator import itemgetter
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import argparse
from torch.nn import CrossEntropyLoss

train_fct = CrossEntropyLoss()
val_fct = CrossEntropyLoss(reduction='none')

class SimCTGPretraining(nn.Module):
    def __init__(self, model_name, use_cl_loss=True, from_scratch=False):
        super(SimCTGPretraining, self).__init__()
        from transformers import AutoTokenizer, GPT2LMHeadModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)        
        if from_scratch: # randomly initialize model
            print ('Pre-training the model from scratch.')
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            self.model = GPT2LMHeadModel(config)
        else:
            print ('Further pre-train with available parameters.')
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        self.embed_dim = self.model.config.hidden_size
        self.logsftmax = nn.LogSoftmax(dim=-1)
        self.use_cl_loss = use_cl_loss

    def compute_logits_and_hidden_states(self, input_ids):
        # used for advanced decoding
        # input_ids: 1 x seqlen
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        return last_hidden_states, logits

    def compute_mle_loss(self, input_ids, labels):
        bsz, seqlen = input_ids.size()
        logits = self.model(input_ids=input_ids, output_hidden_states=True).logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        mle_loss = train_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        return mle_loss

    def compute_contrastive_loss(self, score_matrix, margin):
        '''
           margin: predefined margin to push similarity score away
           score_matrix: bsz x seqlen x seqlen; cosine similarity matrix
           input_ids: bsz x seqlen
        '''
        bsz, seqlen, _ = score_matrix.size()
        gold_score = torch.diagonal(score_matrix, offset=0, dim1=1, dim2=2) # bsz x seqlen
        gold_score = torch.unsqueeze(gold_score, -1)
        assert gold_score.size() == torch.Size([bsz, seqlen, 1])
        difference_matrix = gold_score - score_matrix
        assert difference_matrix.size() == torch.Size([bsz, seqlen, seqlen])
        loss_matrix = margin - difference_matrix # bsz x seqlen x seqlen
        loss_matrix = torch.nn.functional.relu(loss_matrix)
        cl_loss = torch.mean(loss_matrix)
        return cl_loss

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
        if self.use_cl_loss:
            cl_loss = self.compute_contrastive_loss(cosine_scores, margin)
        else:
            cl_loss = torch.Tensor([0.])
            if input_ids.is_cuda:
                cl_loss = cl_loss.cuda(input_ids.get_device())
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

    def parse_output(self, output, eos_token):
        output_text = self.tokenizer.decode(output)
        output_text = output_text.split(eos_token)[0]
        if 'end' in eos_token: # English GPT
            pass
        else: # Chinese GPT
            output_text = ''.join(output_text.strip().split())
        return output_text

    # decoding functions
    # ------------------------------------------------------- #
    def slow_contrastive_search(self, input_ids, beam_width, alpha, decoding_len, eos_token):
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
        #return input_ids[0]
        return self.parse_output(input_ids[0], eos_token)

    def fast_contrastive_search(self, input_ids, beam_width, alpha, decoding_len, eos_token):
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
        #return generated[0]
        return self.parse_output(generated[0], eos_token)

    def diverse_contrastive_search(self, input_ids, sample_step, nucleus_p, beam_width, 
        alpha, decoding_len, eos_token):
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
        output = self.fast_contrastive_search(input_ids, beam_width, alpha, 
            contrastive_step, eos_token)
        return output

    def greedy_search(self, input_ids, decoding_len, eos_token):
        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            max_length=prefix_len+decoding_len)
        #return output[0]
        return self.parse_output(output[0], eos_token)

    def beam_search(self, input_ids, beam_width, decoding_len, eos_token):
        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            max_length=prefix_len+decoding_len, 
                            num_beams=beam_width)
        #return output[0]
        return self.parse_output(output[0], eos_token)


    def nucleus_sampling(self, input_ids, nucleus_p, decoding_len, eos_token):
        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            do_sample=True, 
                            max_length=prefix_len+decoding_len, 
                            top_p=nucleus_p,
                            top_k=0)
        #return output[0]
        return self.parse_output(output[0], eos_token)
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
