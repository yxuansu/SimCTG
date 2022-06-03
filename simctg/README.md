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



