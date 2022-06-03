## Documentation of the SimCTG Library

****
### Catalogue:
* <a href='#simctg_install'>1. Install SimCTG</a>
* <a href='#simctggpt'>2. SimCTGGPT Class</a>
    * <a href='#init_simctggpt'>2.1. Initialization</a>
    * <a href='#forward_simctggpt'>2.2. Forward Computation</a>
* <a href='#simctg_loss'>3. SimCTG Loss Class</a>

****

<span id='simctg_install'/>

#### 1. Install SimCTG:
The package can be easily installed via pip as
```yaml
pip install simctg==0.1
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
* `labels`: The tensor of a bacth labels and its size is bsz x seqlen. The labels is the input_ids right-shifted by one time step. And the padding token is should be replaced -100 to prevent gradient update on padded positions.

You can find an example on how to build the input tensors [[here]](https://github.com/yxuansu/SimCTG#423-create-example-training-data).

:bell: The outputs are as follows:
* `last_hidden_states`: The hidden states of the output layer of the language model and its size is bsz x seqlen x embed_dim.
* `logits`: The output of the prediction linear layer of the language model and its size is bsz x seqlen x vocab_size. The vocab_size = len(model.tokenizer).


****

<span id='simctg_loss'/>

#### 3. SimCTG Loss Class:
To import and initialize the class, run the following command:
```python
from simctg.lossfunction import SimCTGLoss
margin = # the margin in the contrastive loss term
vocab_size = # the vocabulary size of the language model
pad_token_id = # the token id of the padding token 
simctgloss = SimCTGLoss(margin=margin, vocab_size=vocab_size, pad_token_id=pad_token_id)
```



