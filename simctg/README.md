## Documentation of the SimCTG Library

****
### Catalogue:
* <a href='#simctg_install'>1. Install SimCTG</a>
* <a href='#simctg_loss'>2. SimCTGLoss Class</a>
    * <a href='#init_simctgloss'>2.1. Initialization</a>
    * <a href='#forward_simctgloss'>2.2. Forward Computation</a>
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


****

<span id='simctg_install'/>

#### 1. Install SimCTG:
The package can be easily installed via pip as
```yaml
pip install simctg
```

****

<span id='simctg_loss'/>

#### 2. SimCTGLoss Class:

<span id='init_simctgloss'/>

##### 2.1. Initialization:

Initializing the SimCTGLoss class
```python
from simctg.lossfunction import SimCTGLoss
margin = # 
 = # the vocabulary size of the language model
pad_token_id = # the token id of the padding token 
simctgloss = SimCTGLoss(margin=margin, vocab_size=vocab_size, pad_token_id=pad_token_id)
```

:bell: The parameters are as follows:
* `model_name`: The margin in the contrastive loss term (Eq. (2) of our paper).
* `vocab_size`: The vocabulary size of the language model. See more details [[here]]().
* `pad_token_id`: The token id for the padding token. See more details [[here]](https://github.com/yxuansu/SimCTG#422-initialize-loss-class).


<span id='forward_simctgloss'/>

##### 2.2. Forward Computation:



To import and initialize the class, run the following command:
```python
from simctg.lossfunction import SimCTGLoss
margin = # the margin in the contrastive loss term
vocab_size = # the vocabulary size of the language model
pad_token_id = # the token id of the padding token 
simctgloss = SimCTGLoss(margin=margin, vocab_size=vocab_size, pad_token_id=pad_token_id)
```

More details of input refer to below sections.

****

<span id='simctggpt'/>

#### 3. SimCTGGPT Class:

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
* `special_token_list`: The list of user-defined special tokens that are added to the model embedding layer and the tokenizer. It should be a list of tokens, e.g., `["[token_1]", "[token_2]", "[token_3]"]`.


<span id='forward_simctggpt'/>

##### 3.2. Forward Computation:
```python
last_hidden_states, logits = model(input_ids=input_ids, labels=labels)
```

:bell: The inputs are as follows:
* `input_ids`: The tensor of a batch input ids and its size is bsz x seqlen. The tensor should be right-padded with a padding token id.
* `labels`: The tensor of a bacth labels and its size is bsz x seqlen. The labels is the input_ids right-shifted by one time step. And the padding token is should be replaced **-100** to prevent gradient update on padded positions.

You can find an example on how to build the input tensors [[here]](https://github.com/yxuansu/SimCTG#423-create-example-training-data).

:bell: The outputs are as follows:
* `last_hidden_states`: The hidden states of the output layer of the language model and its size is `bsz x seqlen x embed_dim`.
* `logits`: The output of the prediction linear layer of the language model and its size is `bsz x seqlen x vocab_size`. The `vocab_size = len(model.tokenizer)`.

**[Note]** For more detailed definition of last_hidden_states and logits, please refer to the huggingface's documentation [[here]](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel).


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
* `alpha`: The $\alpha$ in contrastive search and $\alpha\in [0.0, 1.0]$ (See Eq. (5) of the paper).
* `decoding_len`: The number of tokens to generate.
* `end_of_sequence_token_id`: The id of the end of sequence token and its default value is `None`:
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
* `alpha`: The $\alpha$ in contrastive search and $\alpha\in [0.0, 1.0]$ (See Eq. (5) of the paper).
* `decoding_len`: The total number of tokens to generate.
* `end_of_sequence_token_id`: The id of the end of sequence token and its default value is `None`:
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
* `end_of_sequence_token_id`: The id of the end of sequence token and its default value is `None`:
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
* `end_of_sequence_token_id`: The id of the end of sequence token and its default value is `None`:
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
* `end_of_sequence_token_id`: The id of the end of sequence token and its default value is `None`:
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
* `end_of_sequence_token_id`: The id of the end of sequence token and its default value is `None`:
* `early_stop`: Whether to truncate the generated output with the end_of_sequence_token_id. The early_stop $\in$ [True, False] and its default value is `False`.

:bell: The output is as follows:
* `output`: A list of output token ids. If `early_stop` is False, then `len(output) = prefix_len + decoding_len`. The output can be easily transformed into the corresponding raw text with `model.tokenizer.decode(output)`.






