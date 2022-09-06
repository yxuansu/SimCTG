## Documentation of the SimCTG Library

****

<span id='catalogue'/>

### Catalogue:
* <a href='#simctg_install'>1. Install SimCTG</a>
* <a href='#simctg_loss'>2. SimCTGLoss Class</a>
    * <a href='#init_simctgloss'>2.1. Initialization</a>
    * <a href='#forward_simctgloss'>2.2. Compute Loss</a>
* <a href='#simctggpt'>3. SimCTGGPT Class</a>
    * <a href='#init_simctggpt'>3.1. Initialization</a>
    * <a href='#forward_simctggpt'>3.2. Forward Computation</a>
    * <a href='#save_simctggpt'>3.3. Save Model</a>
    * <a href='#decoding_simctggpt'>3.4. Decoding Methods</a>
      * <a href='#contrastive_search_simctggpt'>3.4.1. Contrastive Search</a>
      * <a href='#diverse_contrastive_search_simctggpt'>3.4.2. Diverse Contrastive Search</a>
      * <a href='#greedy_search_simctggpt'>3.4.3. Greedy Search</a>
      * <a href='#beam_search_simctggpt'>3.4.4. Beam Search</a>
      * <a href='#nucleus_sampling_simctggpt'>3.4.5. Nucleus Sampling</a>
      * <a href='#topk_sampling_simctggpt'>3.4.6. Top-k Sampling</a>
* <a href='#simctgopt'>4. SimCTGOPT Class</a>
    * <a href='#init_simctgopt'>4.1. Initialization</a>
    * <a href='#forward_simctgopt'>4.2. Forward Computation</a>
    * <a href='#save_simctgopt'>4.3. Save Model</a>
    * <a href='#decoding_simctgopt'>4.4. Decoding Methods</a>
      * <a href='#contrastive_search_simctgopt'>4.4.1. Contrastive Search</a>
      * <a href='#diverse_contrastive_search_simctgopt'>4.4.2. Diverse Contrastive Search</a>
      * <a href='#greedy_search_simctgopt'>4.4.3. Greedy Search</a>
      * <a href='#beam_search_simctgopt'>4.4.4. Beam Search</a>
      * <a href='#nucleus_sampling_simctgopt'>4.4.5. Nucleus Sampling</a>
      * <a href='#topk_sampling_simctgopt'>4.4.6. Top-k Sampling</a>
* <a href='#simctgt5'>5. SimCTGT5 Class</a>
    * <a href='#init_simctgt5'>5.1. Initialization</a>
      * <a href='#init_simctgt5_example_1'>5.1.1. Initialization without Self-Defining Model and Tokenizer</a>
      * <a href='#init_simctgt5_example_2'>5.1.2. Initialization with Self-Defining Model and Tokenizer</a>
    * <a href='#forward_simctgt5'>5.2. Forward Computation</a>
    * <a href='#save_simctgt5'>5.3. Save Model</a>
    * <a href='#decoding_simctgt5'>5.4. Decoding Methods</a>
      * <a href='#contrastive_search_simctgt5'>5.4.1. Contrastive Search</a>
      * <a href='#diverse_contrastive_search_simctgt5'>5.4.2. Diverse Contrastive Search</a>
      * <a href='#greedy_search_simctgt5'>5.4.3. Greedy Search</a>
      * <a href='#beam_search_simctgt5'>5.4.4. Beam Search</a>
      * <a href='#nucleus_sampling_simctgt5'>5.4.5. Nucleus Sampling</a>
* <a href='#evaluation'>6. Evaluation</a>
   * <a href='#reptition_and_diversity'>6.1. Repetition and Diversity</a>


****

<span id='simctg_install'/>

#### 1. Install SimCTG: <a href='#catalogue'>[Back to Top]</a>
The package can be easily installed via pip as
```yaml
pip install simctg --upgrade
```

****

<span id='simctg_loss'/>

#### 2. SimCTGLoss Class: <a href='#catalogue'>[Back to Top]</a>

<span id='init_simctgloss'/>

##### 2.1. Initialization:

Initializing the SimCTGLoss class
```python
from simctg.lossfunction import SimCTGLoss
simctgloss = SimCTGLoss(margin=margin, vocab_size=vocab_size, pad_token_id=pad_token_id)
```

:bell: The parameters are as follows:
* `model_name`: The margin in the contrastive loss term (Eq. (2) of our paper).
* `vocab_size`: The vocabulary size of the language model. See more details [[here]](https://github.com/yxuansu/SimCTG/blob/main/simctg/README.md#32-forward-computation).
* `pad_token_id`: The token id for the padding token. See more details [[here]](https://github.com/yxuansu/SimCTG#422-initialize-loss-class).


<span id='forward_simctgloss'/>

##### 2.2. Compute Loss:
```python
mle_loss, cl_loss = simctgloss(last_hidden_states=last_hidden_states, logits=logits, 
                               input_ids=input_ids, labels=labels)
simctg_loss = mle_loss + cl_loss
```

:bell: The inputs are as follows:
* `last_hidden_states`: The hidden states of the output layer of the language model and its size is `bsz x seqlen x embed_dim`. See more details [[here]](https://github.com/yxuansu/SimCTG/tree/main/simctg#32-forward-computation).
* `logits`: The output of the prediction linear layer of the language model and its size is `bsz x seqlen x vocab_size`. The `vocab_size = len(model.tokenizer)`. See more details [[here]](https://github.com/yxuansu/SimCTG/tree/main/simctg#32-forward-computation).
* `input_ids`: The tensor of a batch input ids and its size is `bsz x seqlen`. The tensor should be right-padded with a padding token id. See more details [[here]](https://github.com/yxuansu/SimCTG/tree/main/simctg#32-forward-computation).
* `labels`: The tensor of a bacth labels and its size is `bsz x seqlen`. The labels is the input_ids right-shifted by one time step. And the padding token is should be replaced **-100** to prevent gradient update on padded positions. See more details [[here]](https://github.com/yxuansu/SimCTG/tree/main/simctg#32-forward-computation).

:bell: The outputs are as follows:
* `mle_loss`: The part of MLE loss (Eq. (1) of our paper).
* `cl_loss`: The part of CL loss (Eq. (2) of our paper).

**[Note]** If the margin is set as 0.0, the CL loss term will be 0.0. Therefore, the SimCTG loss is equivalent to the MLE loss.

****

<span id='simctggpt'/>

#### 3. SimCTGGPT Class: <a href='#catalogue'>[Back to Top]</a>

<span id='init_simctggpt'/>

##### 3.1. Initialization:
Initializing the model and the tokenizer
```python
from simctg.simctggpt import SimCTGGPT
model = SimCTGGPT(model_name=model_name, special_token_list=special_token_list)
tokenizer = model.tokenizer
```

:bell: The parameters are as follows:
* `model_name`: The name of huggingface pre-trained model.
* `special_token_list`: The list of user-defined special tokens that are added to the model embedding layer and the tokenizer. It should be a list of tokens, e.g., `["[token_1]", "[token_2]", "[token_3]"]`. The default value of `special_token_list` is an empty list `[]`.


<span id='forward_simctggpt'/>

##### 3.2. Forward Computation:
```python
last_hidden_states, logits = model(input_ids=input_ids, labels=labels)
```

:bell: The inputs are as follows:
* `input_ids`: The tensor of a batch input ids and its size is `bsz x seqlen`. The tensor should be right-padded with a padding token id.
* `labels`: The tensor of a bacth labels and its size is `bsz x seqlen`. The labels is the input_ids right-shifted by one time step. And the padding token is should be replaced **-100** to prevent gradient update on padded positions.

You can find an example on how to build the input tensors [[here]](https://github.com/yxuansu/SimCTG#423-create-example-training-data).

:bell: The outputs are as follows:
* `last_hidden_states`: The hidden states of the output layer of the language model and its size is `bsz x seqlen x embed_dim`.
* `logits`: The output of the prediction linear layer of the language model and its size is `bsz x seqlen x vocab_size`. The `vocab_size = len(model.tokenizer)`.

**[Note]** For more detailed definition of `last_hidden_states` and `logits`, please refer to the huggingface's documentation [[here]](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel).


<span id='save_simctggpt'/>

##### 3.3. Save Model:
To save the model, please run the following command:
```python
model.save_model(ckpt_save_path=ckpt_save_path)
```

:bell: The parameter is as follows:
* `ckpt_save_path`: The directory to save the model parameters and the tokenizer. The directory will be automatically created if it does not exist before saving the model.

<span id='decoding_simctggpt'/>

##### 3.4. Decoding Methods:
In the following, we illustrate how to use SimCTG to generate text with diffferent decoding methods.

<span id='contrastive_search_simctggpt'/>

###### 3.4.1. Contrastive Search:
```python
output = model.fast_contrastive_search(input_ids=input_ids, beam_width=beam_width, alpha=alpha, decoding_len=decoding_len,           
                                       end_of_sequence_token_id=end_of_sequence_token_id, early_stop=early_stop)
```

:bell: The inputs are as follows:
* `input_ids`: The token ids of the prefix text with size of `1 x prefix_len`.
* `beam_width`: The $k$ in contrastive search (See Eq. (5) of the paper).
* `alpha`: The $\alpha$ in contrastive search and its range is within [0.0, 1.0] (See Eq. (5) of the paper).
* `decoding_len`: The number of tokens to generate.
* `end_of_sequence_token_id`: The id of the end of sequence token and its default value is `None`.
* `early_stop`: Whether to truncate the generated output with the end_of_sequence_token_id. The early_stop $\in$ [True, False] and its default value is `False`.

:bell: The output is as follows:
* `output`: A list of output token ids. If `early_stop` is False, then `len(output) = prefix_len + decoding_len`. The output can be easily transformed into the corresponding raw text with `model.tokenizer.decode(output)`.


<span id='diverse_contrastive_search_simctggpt'/>

###### 3.4.2. Diverse Contrastive Search:
We can also incorporate a certain level of stochasticity into the decoding process of contrastive search by combining nucleus sampling with contrastive search. For instance, if we would like to generate 128 tokens, we can first use nucleus sampling to generate the first two tokens. Then, for the remaining 126 tokens, we switch back to the contrastive search method. For more details, please refer to `Section 7` and `Appendix I` of our paper.

The implementation of diverse contrastive search is as follows:
```python
output = model.diverse_contrastive_search(input_ids=input_ids, sample_step=sample_step, nucleus_p=nucleus_p, 
                                          beam_width=beam_width, alpha=alpha, decoding_len=decoding_len,           
                                          end_of_sequence_token_id=end_of_sequence_token_id, early_stop=early_stop)
```

:bell: The inputs are as follows:
* `input_ids`: The token ids of the prefix text with size of `1 x prefix_len`.
* `sample_step`: The number of tokens that we generate with nucleus sampling at the **start** of the generation process.
* `nucleus_p`: The probability $p$ of nuclues sampling.
* `beam_width`: The $k$ in contrastive search (See Eq. (5) of the paper).
* `alpha`: The $\alpha$ in contrastive search and its range is within [0.0, 1.0] (See Eq. (5) of the paper).
* `decoding_len`: The total number of tokens to generate.
* `end_of_sequence_token_id`: The id of the end of sequence token and its default value is `None`.
* `early_stop`: Whether to truncate the generated output with the end_of_sequence_token_id. The early_stop $\in$ [True, False] and its default value is `False`.

:bell: The output is as follows:
* `output`: A list of output token ids. If `early_stop` is False, then `len(output) = prefix_len + decoding_len`. The output can be easily transformed into the corresponding raw text with `model.tokenizer.decode(output)`.

<span id='greedy_search_simctggpt'/>

###### 3.4.3. Greedy Search:
```python
output = model.greedy_search(input_ids=input_ids, decoding_len=decoding_len,           
                             end_of_sequence_token_id=end_of_sequence_token_id, early_stop=early_stop)
```

:bell: The inputs are as follows:
* `input_ids`: The token ids of the prefix text with size of `1 x prefix_len`.
* `decoding_len`: The number of tokens to generate.
* `end_of_sequence_token_id`: The id of the end of sequence token and its default value is `None`.
* `early_stop`: Whether to truncate the generated output with the end_of_sequence_token_id. The early_stop $\in$ [True, False] and its default value is `False`.

:bell: The output is as follows:
* `output`: A list of output token ids. If `early_stop` is False, then `len(output) = prefix_len + decoding_len`. The output can be easily transformed into the corresponding raw text with `model.tokenizer.decode(output)`.

<span id='beam_search_simctggpt'/>

###### 3.4.4. Beam Search:
```python
output = model.beam_search(input_ids=input_ids, beam_width=beam_width, decoding_len=decoding_len,           
                             end_of_sequence_token_id=end_of_sequence_token_id, early_stop=early_stop)
```

:bell: The inputs are as follows:
* `input_ids`: The token ids of the prefix text with size of `1 x prefix_len`.
* `beam_width`: The beam width of beam search.
* `decoding_len`: The number of tokens to generate.
* `end_of_sequence_token_id`: The id of the end of sequence token and its default value is `None`.
* `early_stop`: Whether to truncate the generated output with the end_of_sequence_token_id. The early_stop $\in$ [True, False] and its default value is `False`.

:bell: The output is as follows:
* `output`: A list of output token ids. If `early_stop` is False, then `len(output) = prefix_len + decoding_len`. The output can be easily transformed into the corresponding raw text with `model.tokenizer.decode(output)`.


<span id='nucleus_sampling_simctggpt'/>

###### 3.4.5. Nucleus Sampling:
```python
output = model.nucleus_sampling(input_ids=input_ids, nucleus_p=nucleus_p, decoding_len=decoding_len,           
                             end_of_sequence_token_id=end_of_sequence_token_id, early_stop=early_stop)
```

:bell: The inputs are as follows:
* `input_ids`: The token ids of the prefix text with size of `1 x prefix_len`.
* `nucleus_p`: The probability $p$ in nucleus sampling.
* `decoding_len`: The number of tokens to generate.
* `end_of_sequence_token_id`: The id of the end of sequence token and its default value is `None`.
* `early_stop`: Whether to truncate the generated output with the end_of_sequence_token_id. The early_stop $\in$ [True, False] and its default value is `False`.

:bell: The output is as follows:
* `output`: A list of output token ids. If `early_stop` is False, then `len(output) = prefix_len + decoding_len`. The output can be easily transformed into the corresponding raw text with `model.tokenizer.decode(output)`.

<span id='topk_sampling_simctggpt'/>

###### 3.4.6. Top-k Sampling:
```python
output = model.topk_sampling(input_ids=input_ids, topk=topk, decoding_len=decoding_len,           
                             end_of_sequence_token_id=end_of_sequence_token_id, early_stop=early_stop)
```

:bell: The inputs are as follows:
* `input_ids`: The token ids of the prefix text with size of `1 x prefix_len`.
* `topk`: The $k$ in top-k sampling.
* `decoding_len`: The number of tokens to generate.
* `end_of_sequence_token_id`: The id of the end of sequence token and its default value is `None`.
* `early_stop`: Whether to truncate the generated output with the end_of_sequence_token_id. The early_stop $\in$ [`True`, `False`] and its default value is `False`.

:bell: The output is as follows:
* `output`: A list of output token ids. If `early_stop` is False, then `len(output) = prefix_len + decoding_len`. The output can be easily transformed into the corresponding raw text with `model.tokenizer.decode(output)`.



****

<span id='simctgopt'/>

#### 4. SimCTGOPT Class: <a href='#catalogue'>[Back to Top]</a>

<span id='init_simctgopt'/>

##### 4.1. Initialization:
Initializing the model and the tokenizer
```python
from simctg.simctgopt import SimCTGOPT
model = SimCTGGPT(model_name=model_name, special_token_list=special_token_list)
tokenizer = model.tokenizer
```

:bell: The parameters are as follows:
* `model_name`: The name of huggingface pre-trained model.
* `special_token_list`: The list of user-defined special tokens that are added to the model embedding layer and the tokenizer. It should be a list of tokens, e.g., `["[token_1]", "[token_2]", "[token_3]"]`. The default value of `special_token_list` is an empty list `[]`.


<span id='forward_simctgopt'/>

##### 4.2. Forward Computation:
```python
last_hidden_states, logits = model(input_ids=input_ids, labels=labels)
```

:bell: The inputs are as follows:
* `input_ids`: The tensor of a batch input ids and its size is `bsz x seqlen`. The tensor should be right-padded with a padding token id.
* `labels`: The tensor of a bacth labels and its size is `bsz x seqlen`. The labels is the input_ids right-shifted by one time step. And the padding token is should be replaced **-100** to prevent gradient update on padded positions.

You can find an example on how to build the input tensors [[here]](https://github.com/yxuansu/SimCTG#423-create-example-training-data).

:bell: The outputs are as follows:
* `last_hidden_states`: The hidden states of the output layer of the language model and its size is `bsz x seqlen x embed_dim`.
* `logits`: The output of the prediction linear layer of the language model and its size is `bsz x seqlen x vocab_size`. The `vocab_size = len(model.tokenizer)`.

**[Note]** For more detailed definition of `last_hidden_states` and `logits`, please refer to the huggingface's documentation [[here]](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel).


<span id='save_simctgopt'/>

##### 4.3. Save Model:
To save the model, please run the following command:
```python
model.save_model(ckpt_save_path=ckpt_save_path)
```

:bell: The parameter is as follows:
* `ckpt_save_path`: The directory to save the model parameters and the tokenizer. The directory will be automatically created if it does not exist before saving the model.

<span id='decoding_simctgopt'/>

##### 4.4. Decoding Methods:
In the following, we illustrate how to use SimCTG to generate text with diffferent decoding methods.

<span id='contrastive_search_simctgopt'/>

###### 4.4.1. Contrastive Search:
```python
output = model.fast_contrastive_search(input_ids=input_ids, beam_width=beam_width, alpha=alpha, decoding_len=decoding_len,           
                                       end_of_sequence_token_id=end_of_sequence_token_id, early_stop=early_stop)
```

:bell: The inputs are as follows:
* `input_ids`: The token ids of the prefix text with size of `1 x prefix_len`.
* `beam_width`: The $k$ in contrastive search (See Eq. (5) of the paper).
* `alpha`: The $\alpha$ in contrastive search and its range is within [0.0, 1.0] (See Eq. (5) of the paper).
* `decoding_len`: The number of tokens to generate.
* `end_of_sequence_token_id`: The id of the end of sequence token and its default value is `None`.
* `early_stop`: Whether to truncate the generated output with the end_of_sequence_token_id. The early_stop $\in$ [True, False] and its default value is `False`.

:bell: The output is as follows:
* `output`: A list of output token ids. If `early_stop` is False, then `len(output) = prefix_len + decoding_len`. The output can be easily transformed into the corresponding raw text with `model.tokenizer.decode(output)`.


<span id='diverse_contrastive_search_simctgopt'/>

###### 4.4.2. Diverse Contrastive Search:
We can also incorporate a certain level of stochasticity into the decoding process of contrastive search by combining nucleus sampling with contrastive search. For instance, if we would like to generate 128 tokens, we can first use nucleus sampling to generate the first two tokens. Then, for the remaining 126 tokens, we switch back to the contrastive search method. For more details, please refer to `Section 7` and `Appendix I` of our paper.

The implementation of diverse contrastive search is as follows:
```python
output = model.diverse_contrastive_search(input_ids=input_ids, sample_step=sample_step, nucleus_p=nucleus_p, 
                                          beam_width=beam_width, alpha=alpha, decoding_len=decoding_len,           
                                          end_of_sequence_token_id=end_of_sequence_token_id, early_stop=early_stop)
```

:bell: The inputs are as follows:
* `input_ids`: The token ids of the prefix text with size of `1 x prefix_len`.
* `sample_step`: The number of tokens that we generate with nucleus sampling at the **start** of the generation process.
* `nucleus_p`: The probability $p$ of nuclues sampling.
* `beam_width`: The $k$ in contrastive search (See Eq. (5) of the paper).
* `alpha`: The $\alpha$ in contrastive search and its range is within [0.0, 1.0] (See Eq. (5) of the paper).
* `decoding_len`: The total number of tokens to generate.
* `end_of_sequence_token_id`: The id of the end of sequence token and its default value is `None`.
* `early_stop`: Whether to truncate the generated output with the end_of_sequence_token_id. The early_stop $\in$ [True, False] and its default value is `False`.

:bell: The output is as follows:
* `output`: A list of output token ids. If `early_stop` is False, then `len(output) = prefix_len + decoding_len`. The output can be easily transformed into the corresponding raw text with `model.tokenizer.decode(output)`.

<span id='greedy_search_simctgopt'/>

###### 4.4.3. Greedy Search:
```python
output = model.greedy_search(input_ids=input_ids, decoding_len=decoding_len,           
                             end_of_sequence_token_id=end_of_sequence_token_id, early_stop=early_stop)
```

:bell: The inputs are as follows:
* `input_ids`: The token ids of the prefix text with size of `1 x prefix_len`.
* `decoding_len`: The number of tokens to generate.
* `end_of_sequence_token_id`: The id of the end of sequence token and its default value is `None`.
* `early_stop`: Whether to truncate the generated output with the end_of_sequence_token_id. The early_stop $\in$ [True, False] and its default value is `False`.

:bell: The output is as follows:
* `output`: A list of output token ids. If `early_stop` is False, then `len(output) = prefix_len + decoding_len`. The output can be easily transformed into the corresponding raw text with `model.tokenizer.decode(output)`.

<span id='beam_search_simctgopt'/>

###### 4.4.4. Beam Search:
```python
output = model.beam_search(input_ids=input_ids, beam_width=beam_width, decoding_len=decoding_len,           
                             end_of_sequence_token_id=end_of_sequence_token_id, early_stop=early_stop)
```

:bell: The inputs are as follows:
* `input_ids`: The token ids of the prefix text with size of `1 x prefix_len`.
* `beam_width`: The beam width of beam search.
* `decoding_len`: The number of tokens to generate.
* `end_of_sequence_token_id`: The id of the end of sequence token and its default value is `None`.
* `early_stop`: Whether to truncate the generated output with the end_of_sequence_token_id. The early_stop $\in$ [True, False] and its default value is `False`.

:bell: The output is as follows:
* `output`: A list of output token ids. If `early_stop` is False, then `len(output) = prefix_len + decoding_len`. The output can be easily transformed into the corresponding raw text with `model.tokenizer.decode(output)`.


<span id='nucleus_sampling_simctgopt'/>

###### 4.4.5. Nucleus Sampling:
```python
output = model.nucleus_sampling(input_ids=input_ids, nucleus_p=nucleus_p, decoding_len=decoding_len,           
                             end_of_sequence_token_id=end_of_sequence_token_id, early_stop=early_stop)
```

:bell: The inputs are as follows:
* `input_ids`: The token ids of the prefix text with size of `1 x prefix_len`.
* `nucleus_p`: The probability $p$ in nucleus sampling.
* `decoding_len`: The number of tokens to generate.
* `end_of_sequence_token_id`: The id of the end of sequence token and its default value is `None`.
* `early_stop`: Whether to truncate the generated output with the end_of_sequence_token_id. The early_stop $\in$ [True, False] and its default value is `False`.

:bell: The output is as follows:
* `output`: A list of output token ids. If `early_stop` is False, then `len(output) = prefix_len + decoding_len`. The output can be easily transformed into the corresponding raw text with `model.tokenizer.decode(output)`.

<span id='topk_sampling_simctgopt'/>

###### 4.4.6. Top-k Sampling:
```python
output = model.topk_sampling(input_ids=input_ids, topk=topk, decoding_len=decoding_len,           
                             end_of_sequence_token_id=end_of_sequence_token_id, early_stop=early_stop)
```

:bell: The inputs are as follows:
* `input_ids`: The token ids of the prefix text with size of `1 x prefix_len`.
* `topk`: The $k$ in top-k sampling.
* `decoding_len`: The number of tokens to generate.
* `end_of_sequence_token_id`: The id of the end of sequence token and its default value is `None`.
* `early_stop`: Whether to truncate the generated output with the end_of_sequence_token_id. The early_stop $\in$ [`True`, `False`] and its default value is `False`.

:bell: The output is as follows:
* `output`: A list of output token ids. If `early_stop` is False, then `len(output) = prefix_len + decoding_len`. The output can be easily transformed into the corresponding raw text with `model.tokenizer.decode(output)`.

****

<span id='simctgt5'/>

#### 5. SimCTGT5 Class: <a href='#catalogue'>[Back to Top]</a>

<span id='init_simctgt5'/>

##### 5.1. Initialization:
Initializing the model and the tokenizer
```python
from simctg.simctgt5 import SimCTGT5
model = SimCTGT5(model_name=model_name, user_defined_model=self_defined_model, user_defined_tokenizer=self_defined_tokenizer, special_token_list=special_token_list)
tokenizer = model.tokenizer
```

:bell: The parameters are as follows:
* `model_name`: The name of huggingface pre-trained model.
* `user_defined_model`: The T5 model self-defined by the user (possibly not publically available). The default value of `user_defined_model` is `None`.
* `user_defined_tokenizer`: The tokenizer self-defined by the user (possibly not publically available). The default value of `user_defined_tokenizer` is `None`.
* `special_token_list`: The list of user-defined special tokens that are added to the model embedding layer and the tokenizer. It should be a list of tokens, e.g., `["[token_1]", "[token_2]", "[token_3]"]`. The default value of `special_token_list` is an empty list `[]`.

Below are two examples of how to initialize the model.

<span id='init_simctgt5_example_1'/>

###### 5.1.1. Initialization without Self-Defining Model and Tokenizer:
```python
from simctg.simctgt5 import SimCTGT5
model_name = "flax-community/t5-base-cnn-dm"
model = SimCTGT5(model_name, special_token_list=[])
```

<span id='init_simctgt5_example_2'/>

###### 5.1.2. Initialization with Self-Defining Model and Tokenizer:
```python
from simctg.simctgt5 import SimCTGT5
model_name = r'imxly/t5-pegasus'
# define tokenizer
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
# define model
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
t5model = MT5ForConditionalGeneration.from_pretrained(model_name)
# initialization
model = SimCTGT5(model_name, user_defined_model=t5model, user_defined_tokenizer=tokenizer, special_token_list=[])
```

<span id='forward_simctgt5'/>

##### 5.2. Forward Computation:

```python
last_hidden_states, logits = model(encoder_inputs=encoder_inputs, encoder_mask=encoder_mask, 
                                   decoder_inputs=decoder_inputs, decoder_labels=decoder_labels)
```

:bell: The inputs are as follows:
* `encoder_inputs`: The tensor of a batch input ids on the encoder side and its size is `bsz x src_len`. The tensor should be right-padded with a padding token id.
* `encoder_mask`: Mask to avoid performing attention on padding token indices on the encoder side. Mask values selected in [0, 1]: (i) 1 for tokens that are not masked; and (ii) 0 for tokens that are masked. Its size is `bsz x src_len`.
* `decoder_inputs`: The tensor of a batch input ids on the decoder side and its size is `bsz x tgt_len`. The tensor should be right-padded with a padding token id.
* `decoder_labels`: The tensor of a bacth labels on the decoder side and its size is `bsz x tgt_len`. The labels is the `decoder_inputs` right-shifted by one time step. And the padding token is should be replaced **-100** to prevent gradient update on padded positions.

:bell: The outputs are as follows:
* `last_hidden_states`: The hidden states of the output layer of the decoder and its size is `bsz x tgt_len x embed_dim`.
* `logits`: The output of the prediction linear layer of the model and its size is `bsz x tgt_len x vocab_size`. The `vocab_size = len(model.tokenizer)`.

**[Note]** For more detailed definition of `last_hidden_states` and `logits`, please refer to the huggingface's documentation [[here]](https://huggingface.co/docs/transformers/model_doc/t5#transformers.TFT5ForConditionalGeneration).


<span id='save_simctgt5'/>

##### 5.3. Save Model:
To save the model, please run the following command:
```python
model.save_model(ckpt_save_path=ckpt_save_path)
```

:bell: The parameter is as follows:
* `ckpt_save_path`: The directory to save the model parameters and the tokenizer. The directory will be automatically created if it does not exist before saving the model.


<span id='decoding_simctgt5'/>

##### 5.4. Decoding Methods:

In the following, we illustrate how to generate text with SimCTGT5.

<span id='contrastive_search_simctgt5'/>

###### 5.4.1. Contrastive Search:
```python
output = model.fast_contrastive_search(input_ids=input_ids, beam_width=beam_width, alpha=alpha, decoding_len=decoding_len, 
                                       start_of_sequence_token_id=start_of_sequence_token_id, 
                                       end_of_sequence_token_id=end_of_sequence_token_id, early_stop=early_stop)
```

:bell: The inputs are as follows:
* `input_ids`: The input token ids of the encoder with size of `1 x src_len`.
* `beam_width`: The $k$ in contrastive search.
* `alpha`: The $\alpha$ in contrastive search and its range is within [0.0, 1.0].
* `decoding_len`: The number of tokens to generate.
* `start_of_sequence_token_id`: The start token id of the decoder to start generation. If it is set as `None`, then we use the default start token id. Otherwise, the user can self-define the start token id of the model. The default value of this argument is `None`.
* `end_of_sequence_token_id`: The end token id of the decoder that indicates the end of generation. If it is set as `None`, then we use the default end token id of the model. Otherwise, the user can self-define the end token id. The default value of this argument is `None`.
* `early_stop`: Whether to truncate and early-stop the generated output with the end_of_sequence_token_id. The early_stop $\in$ [True, False] and its default value is `True`.

:bell: The output is as follows:
* `output`: A list of output token ids. The output can be easily transformed into the corresponding raw text with `model.tokenizer.decode(output)`.

**[Examples]** Two example usages of contrastive search can be found [[here]](https://github.com/yxuansu/SimCTG/tree/main/SimCTGEncDec#22-contrastive-search) and [[here]](https://github.com/yxuansu/SimCTG/issues/5#issuecomment-1163309924).


<span id='diverse_contrastive_search_simctgt5'/>

###### 5.4.2. Diverse Contrastive Search:
**[Definition]** The definition of diverse contrastive search can be found <a href='#diverse_contrastive_search_simctggpt'>[here]</a>.
```python
output = model.diverse_contrastive_search(input_ids=input_ids, sample_step=sample_step, nucleus_p=nucleus_p, beam_width=beam_width, 
                                          alpha=alpha, decoding_len=decoding_len, start_of_sequence_token_id=start_of_sequence_token_id,
                                          end_of_sequence_token_id=end_of_sequence_token_id, early_stop=early_stop)
```

:bell: The inputs are as follows:
* `input_ids`: The input token ids of the encoder with size of `1 x src_len`.
* `sample_step`: The number of tokens that we generate with nucleus sampling at the **start** of the generation process.
* `nucleus_p`: The probability $p$ of nuclues sampling.
* `beam_width`: The $k$ in contrastive search.
* `alpha`: The $\alpha$ in contrastive search and its range is within [0.0, 1.0].
* `decoding_len`: The number of tokens to generate.
* `start_of_sequence_token_id`: The start token id of the decoder to start generation. If it is set as `None`, then we use the default start token id. Otherwise, the user can self-define the start token id of the model. The default value of this argument is `None`.
* `end_of_sequence_token_id`: The end token id of the decoder that indicates the end of generation. If it is set as `None`, then we use the default end token id of the model. Otherwise, the user can self-define the end token id. The default value of this argument is `None`.
* `early_stop`: Whether to truncate and early-stop the generated output with the end_of_sequence_token_id. The early_stop $\in$ [True, False] and its default value is `True`.

:bell: The output is as follows:
* `output`: A list of output token ids. The output can be easily transformed into the corresponding raw text with `model.tokenizer.decode(output)`.

**[Example]** One example usage of diverse contrastive search can be found [[here]](https://github.com/yxuansu/SimCTG/tree/main/SimCTGEncDec#23-diverse-contrastive-search).

<span id='greedy_search_simctgt5'/>

###### 5.4.3. Greedy Search:
```python
output = model.greedy_search(input_ids=input_ids, decoding_len=decoding_len, start_of_sequence_token_id=start_of_sequence_token_id, 
                             end_of_sequence_token_id=end_of_sequence_token_id, early_stop=early_stop)
```

:bell: The inputs are as follows:
* `input_ids`: The input token ids of the encoder with size of `1 x src_len`.
* `decoding_len`: The number of tokens to generate.
* `start_of_sequence_token_id`: The start token id of the decoder to start generation. If it is set as `None`, then we use the default start token id. Otherwise, the user can self-define the start token id of the model. The default value of this argument is `None`.
* `end_of_sequence_token_id`: The end token id of the decoder that indicates the end of generation. If it is set as `None`, then we use the default end token id of the model. Otherwise, the user can self-define the end token id. The default value of this argument is `None`.
* `early_stop`: Whether to truncate and early-stop the generated output with the end_of_sequence_token_id. The early_stop $\in$ [True, False] and its default value is `True`.

:bell: The output is as follows:
* `output`: A list of output token ids. The output can be easily transformed into the corresponding raw text with `model.tokenizer.decode(output)`.

**[Example]** One example usage of greedy search can be found [[here]](https://github.com/yxuansu/SimCTG/blob/main/SimCTGEncDec/README.md#24-greedy-search).

<span id='beam_search_simctgt5'/>

###### 5.4.4. Beam Search:
```python
output = model.beam_search(input_ids=input_ids, beam_width=beam_width, decoding_len=decoding_len, 
                           start_of_sequence_token_id=start_of_sequence_token_id, 
                           end_of_sequence_token_id=end_of_sequence_token_id, early_stop=early_stop)
```

:bell: The inputs are as follows:
* `input_ids`: The input token ids of the encoder with size of `1 x src_len`.
* `beam_width`: The beam width of beam search.
* `decoding_len`: The number of tokens to generate.
* `start_of_sequence_token_id`: The start token id of the decoder to start generation. If it is set as `None`, then we use the default start token id. Otherwise, the user can self-define the start token id of the model. The default value of this argument is `None`.
* `end_of_sequence_token_id`: The end token id of the decoder that indicates the end of generation. If it is set as `None`, then we use the default end token id of the model. Otherwise, the user can self-define the end token id. The default value of this argument is `None`.
* `early_stop`: Whether to truncate and early-stop the generated output with the end_of_sequence_token_id. The early_stop $\in$ [True, False] and its default value is `True`.

:bell: The output is as follows:
* `output`: A list of output token ids. The output can be easily transformed into the corresponding raw text with `model.tokenizer.decode(output)`.

**[Example]** One example usage of beam search can be found [[here]](https://github.com/yxuansu/SimCTG/blob/main/SimCTGEncDec/README.md#25-beam-search).

<span id='nucleus_sampling_simctgt5'/>

###### 5.4.5. Nucleus Sampling:
```python
output = model.nucleus_sampling(input_ids=input_ids, nucleus_p=nucleus_p, decoding_len=decoding_len, 
                           start_of_sequence_token_id=start_of_sequence_token_id, 
                           end_of_sequence_token_id=end_of_sequence_token_id, early_stop=early_stop)
```

:bell: The inputs are as follows:
* `input_ids`: The input token ids of the encoder with size of `1 x src_len`.
* `nucleus_p`: The probability $p$ of nuclues sampling.
* `decoding_len`: The number of tokens to generate.
* `start_of_sequence_token_id`: The start token id of the decoder to start generation. If it is set as `None`, then we use the default start token id. Otherwise, the user can self-define the start token id of the model. The default value of this argument is `None`.
* `end_of_sequence_token_id`: The end token id of the decoder that indicates the end of generation. If it is set as `None`, then we use the default end token id of the model. Otherwise, the user can self-define the end token id. The default value of this argument is `None`.
* `early_stop`: Whether to truncate and early-stop the generated output with the end_of_sequence_token_id. The early_stop $\in$ [True, False] and its default value is `True`.

:bell: The output is as follows:
* `output`: A list of output token ids. The output can be easily transformed into the corresponding raw text with `model.tokenizer.decode(output)`.

**[Example]** One example usage of nucleus sampling can be found [[here]](https://github.com/yxuansu/SimCTG/blob/main/SimCTGEncDec/README.md#26-nucleus-sampling).




****

<span id='evaluation'/>

#### 6. Evaluation: <a href='#catalogue'>[Back to Top]</a>

<span id='reptition_and_diversity'/>

##### 6.1. Repetition and Diversity:
Here, we show how to replicate the n-gram repetition and diversity results of contrastive search as reported in the paper.

(1) First, download the prediction result of contrastive search as provided in our repo [[here]](https://github.com/yxuansu/SimCTG/blob/main/document_generation/simctg_contrasive.json).
```yaml
wget https://raw.githubusercontent.com/yxuansu/SimCTG/main/document_generation/simctg_contrasive.json
```

(2) Second, replicate the n-gram repetition and diversity results as:
```python
# parse the generated results into a list of text
import json
in_f = r'./simctg_contrasive.json'
with open(in_f) as f:
    item_list = json.load(f)

text_list = []
for item in item_list:
    text = item['generated_result']['0']['continuation']
    text_list.append(text)

# compute the evaluation results
from simctg.evaluation import measure_repetition_and_diversity
rep_2, rep_3, rep_4, diversity = measure_repetition_and_diversity(text_list)
print ('The result of rep-2 is {}, rep-3 is {}, rep-4 is {}, and diversity is {}'.format(rep_2, rep_3, rep_4, round(diversity,2)))
'''
   The result of rep-2 is 3.93, rep-3 is 0.78, rep-4 is 0.31, and diversity is 0.95
'''
```

The input to the function `measure_repetition_and_diversity()` is a list of text and it returns the results of rep-2, rep-3, rep-4, and diversity. The definitions of different metrics are

(i) $\textup{\textbf{rep-n}}=100 \times (1.0 - \frac{|\textup{unique n-grams}(\hat{\boldsymbol{x}})|}{|\textup{total n-grams}(\hat{\boldsymbol{x}})|})$ 

(ii) $\textup{\textbf{diversity}}=(1.0-\frac{\textup{rep-2}}{100})\times (1.0-\frac{\textup{rep-3}}{100}) \times (1.0-\frac{\textup{rep-4}}{100})$.






