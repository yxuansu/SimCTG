import sys
import ipdb
import os
import operator
from operator import itemgetter
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import argparse
import random

def ranking(context_hidden, next_hidden, next_top_k_ids, next_top_k_probs, alpha):
    '''
        context_hidden: beam_width x context_len x embed_dim
        next_hidden: beam_width x 1 x embed_dim
        next_top_k_ids: beam_width x 1
    '''
    beam_width, context_len, embed_dim = context_hidden.size()
    assert next_hidden.size() == torch.Size([beam_width, 1, embed_dim])
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)
    assert cosine_matrix.size() == torch.Size([beam_width, context_len])
    scores, _ = torch.max(cosine_matrix, dim = -1)
    assert scores.size() == torch.Size([beam_width])
    next_top_k_probs = next_top_k_probs.view(-1)
    scores = (1.0 - alpha) * next_top_k_probs - alpha * scores 
    _, selected_idx = torch.topk(scores, k = 1)
    assert selected_idx.size() == torch.Size([1])
    selected_idx = selected_idx.unsqueeze(0)
    assert selected_idx.size() == torch.Size([1,1])
    next_id = torch.gather(next_top_k_ids, dim = 0, index=selected_idx)
    assert next_id.size() == torch.Size([1,1])
    return next_id


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
    first_step=False,
    input_embeds=None,
    ):
    # input_ids: [B, S]
    if first_step:
        output = model(
            input_ids=ids, 
            decoder_input_ids=decoder_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )
        past_key_values = output.past_key_values
        last_hidden_states = output.decoder_hidden_states[-1]    # [B, S, E]
        logit_for_next_step = output.logits[:, -1, :]    # [B, V]
        input_embeds = output.encoder_hidden_states[0]    # [B, S, E]
        input_embeds = input_embeds.expand(beam_width, -1, -1)    # [5, S, E]
    bsz, seqlen, embed_dim = last_hidden_states.size()

    next_probs = F.softmax(logit_for_next_step, dim=-1)
    _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width)    # [B, K]
    top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)    # [B, K]
    # compute new hidden
    past_key_values = encdec_enlarge_past_key_values(past_key_values, beam_width)
    output = model(
        input_ids=torch.cat([ids for _ in range(beam_width)]),
        # inputs_embeds=input_embeds,
        decoder_input_ids=top_k_ids.contiguous().view(-1, 1), 
        past_key_values=past_key_values,
        output_hidden_states=True,
        use_cache=True,
    )
    past_key_values = output.past_key_values
    logits = output.logits[:, -1, :]    # [B*K, V]
    next_hidden = output.decoder_hidden_states[-1]    # [B*K, 1, E]
    context_hidden = last_hidden_states.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(bsz*beam_width, seqlen, embed_dim)    # [B*K, S, E]

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
    return next_id, past_key_values, last_hidden_states, logits, input_embeds 

def EncDecContrastiveDecodingOneStep(model, input_ids, decoder_input_ids, beam_width, alpha):
    '''
        model: the generation model, e.g., gpt2
        input_ids: 1 x seqlen
    '''
    prev_hidden_states, logits = model.compute_logits_and_hidden_states(input_ids, decoder_input_ids)
    _, seqlen, embed_dim = prev_hidden_states.size()
    _, _, vocab_size = logits.size()
    p = random.uniform(0, 1)

    logit_for_next_step = logits[:,-1,:]
    assert logit_for_next_step.size() == torch.Size([1, vocab_size])

    next_probs = F.softmax(logit_for_next_step, dim = -1)
    assert next_probs.size() == logit_for_next_step.size()

    _, top_k_ids = torch.topk(logit_for_next_step, dim = -1, k = beam_width)
    assert top_k_ids.size() == torch.Size([1, beam_width])
        
    top_k_probs = torch.gather(next_probs, dim = 1, index=top_k_ids)

    assert top_k_probs.size() == top_k_ids.size()
    # compute new hidden 
    expanded_context = [decoder_input_ids for _ in range(beam_width)]
    expanded_context = torch.cat(expanded_context, dim = 0)
    expanded_input_ids = [input_ids for _ in range(beam_width)]
    expanded_input_ids = torch.cat(expanded_input_ids)
    assert expanded_context.size() == torch.Size([beam_width, seqlen])
    top_k_ids = top_k_ids.view(beam_width, 1)
    next_input_ids = torch.cat([expanded_context, top_k_ids], dim = -1)
    assert next_input_ids.size() == torch.Size([beam_width, seqlen+1])
    new_hidden_states, next_logits = model.compute_logits_and_hidden_states(expanded_input_ids, next_input_ids)
    assert new_hidden_states.size() == torch.Size([beam_width, seqlen+1, embed_dim])
    context_hidden = new_hidden_states[:,:seqlen,:]
    assert context_hidden.size() == torch.Size([beam_width, seqlen, embed_dim])
    next_hidden = new_hidden_states[:,seqlen:,:]
    assert next_hidden.size() == torch.Size([beam_width, 1, embed_dim])

    next_id = ranking(context_hidden, next_hidden, top_k_ids, top_k_probs, alpha)       

    next_input_ids = torch.cat([decoder_input_ids, next_id], dim = -1)
    assert next_input_ids.size() == torch.Size([1, seqlen+1])
    return next_input_ids

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

