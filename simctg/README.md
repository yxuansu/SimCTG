## Documentation of the SimCTG Library

****
### Catalogue:
* <a href='#simctg_install'>1. Install SimCTG</a>
* <a href='#simctggpt'>2. SimCTGGPT Class</a>
    * <a href='#init_simctggpt'>2.1. Initialization</a>
    * <a href='#forward_simctggpt'>2.2. Forward Computation</a>
    * <a href='#save_simctggpt'>2.3. Save Model</a>
    * <a href='#decoding_simctggpt'>2.4. Decoding Methods</a>
      * <a href='#contrastive_search_simctggpt'>2.4.1. Contrastive Search</a>
      * <a href='#diverse_contrastive_search_simctggpt'>2.4.2. Diverse Contrastive Search</a>
      * <a href='#greedy_search_simctggpt'>2.4.3. Greedy Search</a>
      * <a href='#beam_search_simctggpt'>2.4.4. Beam Search</a>
      * <a href='#nucleus_sampling_simctggpt'>2.4.5. Nucleus Sampling</a>
      * <a href='#topk_sampling_simctggpt'>2.4.6. Top-k Sampling</a>
* <a href='#simctg_loss'>3. SimCTGLoss Class</a>

****

<span id='simctg_install'/>

#### 1. Install SimCTG:
The package can be easily installed via pip as
```yaml
pip install simctg
```

****

<span id='simctggpt'/>

#### 2. SimCTGGPT Class:

<span id='init_simctggpt'/>

##### 2.1. Initialization:
Initializing the model and the tokenizer
```python
from simctg.simctggpt import SimCTGGPT
model = SimCTGGPT(model_name=model_name, special_token_list=special_token_list)
tokenizer = model.tokenizer
```

:bell: The parameters are as follows:
* `model_name`: The name of huggingface pre-trained model.
* `special_token_list`: The list of user-defined special tokens that are added to the model embedding layer and the tokenizer. It should be a list of tokens, e.g., ["[token_1]", "[token_2]", "[token_3]"].


<span id='forward_simctggpt'/>

##### 2.2. Forward Computation:
```python
last_hidden_states, logits = model(input_ids=input_ids, labels=labels)
```

:bell: The inputs are as follows:
* `input_ids`: The tensor of a batch input ids and its size is bsz x seqlen. The tensor should be right-padded with a padding token id.
* `labels`: The tensor of a bacth labels and its size is bsz x seqlen. The labels is the input_ids right-shifted by one time step. And the padding token is should be replaced **-100** to prevent gradient update on padded positions.

You can find an example on how to build the input tensors [[here]](https://github.com/yxuansu/SimCTG#423-create-example-training-data).

:bell: The outputs are as follows:
* `last_hidden_states`: The hidden states of the output layer of the language model and its size is `bsz x seqlen x embed_dim`.
* `logits`: The output of the prediction linear layer of the language model and its size is `bsz x seqlen x vocab_size`. The vocab_size = len(model.tokenizer).


<span id='save_simctggpt'/>

##### 2.3. Save Model:
To save the model, please run the following command:
```python
model.save_model(ckpt_save_path=ckpt_save_path)
```

:bell: The parameter is as follows:
* `ckpt_save_path`: The directory to save the model parameters and the tokenizer. The directory will be automatically created if it does not exist before saving the model.

<span id='decoding_simctggpt'/>

##### 2.4. Decoding Methods:
In the following, we illustrate how to use SimCTG to generate text with diffferent decoding methods.

<span id='contrastive_search_simctggpt'/>

###### 2.4.1. Contrastive Search:
```python
output = model.fast_contrastive_search(input_ids=input_ids, beam_width=beam_width, alpha=alpha, decoding_len=decoding_len,           
                                       end_of_sequence_token_id=end_of_sequence_token_id, early_stop=early_stop)
```


<span id='diverse_contrastive_search_simctggpt'/>

###### 2.4.2. Diverse Contrastive Search:

<span id='greedy_search_simctggpt'/>

###### 2.4.3. Greedy Search:

<span id='beam_search_simctggpt'/>

###### 2.4.4. Beam Search:

<span id='nucleus_sampling_simctggpt'/>

###### 2.4.5. Nucleus Sampling:

<span id='topk_sampling_simctggpt'/>

###### 2.4.6. Top-k Sampling:



****

<span id='simctg_loss'/>

#### 3. SimCTGLoss Class:
To import and initialize the class, run the following command:
```python
from simctg.lossfunction import SimCTGLoss
margin = # the margin in the contrastive loss term
vocab_size = # the vocabulary size of the language model
pad_token_id = # the token id of the padding token 
simctgloss = SimCTGLoss(margin=margin, vocab_size=vocab_size, pad_token_id=pad_token_id)
```



