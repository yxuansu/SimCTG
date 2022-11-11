# Details of the Contrastive Search Implementions

In this section, we briefly describe the motivation and the details of our proposed contrastive search.



For each step during the autoregressive decoding, contrastive search conduct the following two steps to select the most appropriate token to generate. The overall implementation can be found [here](https://github.com/yxuansu/SimCTG/blob/7d7bac2109752a62ef26ba8abe97bf02b507d1c3/simctg/utlisgpt.py#L31).

1. Candidates collection
   
   First of all, we compute the prefix by the language models and obtain the Top-`k` candidate tokens based on their logits (or we called model confidence in the paper)
   
   ```python
   # compute the logits in this step
   prev_hidden_states, logits = model.compute_logits_and_hidden_states(input_ids)
   _, seqlen, embed_dim = prev_hidden_states.size()
   _, _, vocab_size = logits.size()
   
   logit_for_next_step = logits[:,-1,:]
   assert logit_for_next_step.size() == torch.Size([1, vocab_size])
   
   # normalize with the softmax function
   next_probs = F.softmax(logit_for_next_step, dim = -1)
   assert next_probs.size() == logit_for_next_step.size()
   
   # collecte top-k candidate tokens and their logits (model confidence)
   _, top_k_ids = torch.topk(logit_for_next_step, dim = -1, k = beam_width)
   assert top_k_ids.size() == torch.Size([1, beam_width])
           
   top_k_probs = torch.gather(next_probs, dim = 1, index=top_k_ids)
   assert top_k_probs.size() == top_k_ids.size()
   ```
   
   As shown in the above code, `top_k_ids` contaisn the indexes of Top-`k` candidate tokens, and `top_k_probs` saves their model confidence.

2. Candidate re-ranking
   
   Then, we concatenate all the candidate tokens with the prefix and construct the a small batch, and its batch size is `k`.
   
   ```python
   # concatenate each candidate token with prefix
   expanded_context = [input_ids for _ in range(beam_width)]
   expanded_context = torch.cat(expanded_context, dim = 0)
   assert expanded_context.size() == torch.Size([beam_width, seqlen])
   top_k_ids = top_k_ids.view(beam_width, 1)
   next_input_ids = torch.cat([expanded_context, top_k_ids], dim = -1)
   assert next_input_ids.size() == torch.Size([beam_width, seqlen+1])
   ```
   
   This small batch is fed into the language models again to get their next step hidden states.
   
   ```python
   # feed these candidates into next round to get their hidden states
   new_hidden_states, next_logits = model.compute_logits_and_hidden_states(next_input_ids)
   assert new_hidden_states.size() == torch.Size([beam_width, seqlen+1, embed_dim])
   context_hidden = new_hidden_states[:,:seqlen,:]
   assert context_hidden.size() == torch.Size([beam_width, seqlen, embed_dim])
   next_hidden = new_hidden_states[:,seqlen:,:]
   assert next_hidden.size() == torch.Size([beam_width, 1, embed_dim])
   ```
   
   The `next_hidden` contains the hidden states of each candidate token, and will be used to calculate the degeneration penalty (maximum cosine similarity with respect to previous tokens `context_hidden`).
   
   ```python
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
   ```
   
   As shown in the above code, the degeneration penalty and the model confidence are added to re-rank these candidate tokens, which follows this formulation:
   
   ![](https://user-images.githubusercontent.com/27548710/192125120-ca0ddb4d-70da-4489-b65d-885a0c8f96fc.png)
   
   The candidate token that has the highest re-ranking scores will be selected as the next token for generation.
