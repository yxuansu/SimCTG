## Documentation of the SimCTG Library

****
### Catalogue:
* <a href='#simctg_install'>1. Install SimCTG</a>
* <a href='#simctggpt'>2. SimCTGGPT Class</a>
    * <a href='#init_simctggpt'>2.1. Initialization</a>
* <a href='#simctg_loss'>3. SimCTG Loss Class</a>

****

<span id='simctg_install'/>

#### 1. Install SimCTG:
The package can be easily installed via pip as
```yaml
pip install simctg==0.1
```

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

The parameters are as follows:
* `model_name`: The name of huggingface pre-trained model.
* `special_token_list`: The list of user-defined special tokens that are added to the model embedding layer and the tokenizer. It should be a list of tokens, e.g., ["[token_1]", "[token_2]", "[token_3]"].

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



