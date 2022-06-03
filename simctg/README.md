## Documentation of the SimCTG Library

****
### Catalogue:
* <a href='#simctg_install'>1. Install SimCTG</a>
* <a href='#simctg_loss'>2. SimCTG Loss Class</a>
* <a href='#simctggpt'>3. SimCTGGPT Class</a>
****

<span id='simctg_install'/>

#### 1. Install SimCTG:
The package can be easily installed via pip as
```yaml
pip install simctg==0.1
```

<span id='simctg_loss'/>

#### 2. SimCTG Loss Class:
To import and initialize the class, run the following command
```python
from simctg.lossfunction import SimCTGLoss
margin = # the margin in the contrastive loss term
vocab_size = # the vocabulary size of the language model
pad_token_id = # the token id of the padding token 
simctgloss = SimCTGLoss(margin=margin, vocab_size=vocab_size, pad_token_id=pad_token_id)
```


<span id='simctg_loss'/>

#### 2. SimCTG Loss Class:
