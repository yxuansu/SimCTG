# A Contrastive Framework for Neural Text Generation
**Authors**: Yixuan Su, Tian Lan, Yan Wang, Lingpeng Kong, Dani Yogatama, and Nigel Collier

Code of our paper: [A Contrastive Framework for Neural Text Generation]()

****

### News:
[2022//] SimCTG is publicly released!

****
## Introduction

****
## Huggingface Models
### 1. Wikitext-103:

****
## Catalogue:
* <a href='#environment_setup'>1. Environment Setup</a>
* <a href='#example_usage'>2. Example Usage of Contrastive Search</a>
* <a href='#wikitext103_tutorial'>3. Document Generation</a>
* <a href='#dialogue_tutorial'>4. Dialogue Generation>


****

<span id='environment_setup'/>

#### 1. Environment Setup:
```yaml
python version: 3.8
pip3 install -r requirements.txt
```

<span id='example_usage'/>

#### 2. Example Usage of Contrastive Search:
Here, we show how to reproduce the result in Table 3 of our paper.
```python
import torch
from simctg import SimCTG
from transformers import AutoTokenizer
# load model and tokenizer
model_path = r'cambridgeltl/simctg_wikitext103'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = SimCTG(model_path, tokenizer.pad_token_id)
model.eval()

# prepare prefix input
text = r"Butt criticized Donald 's controls in certain situations in the game , as well as the difficulty of some levels and puzzles . Buchanan also criticized the controls , calling"
tokens = tokenizer.tokenize(text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor(input_ids).view(1,-1)

# use contrastive search to generate the result
beam_width, alpha, decoding_len = 8, 0.6, 128
output = model.fast_contrastive_search(input_ids, beam_width, alpha, decoding_len)
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output))

# use diverse contrastive search to generate the result
sample_step, nucleus_p = 2, 0.95
beam_width, alpha, decoding_len = 8, 0.6, 128
output = model.diverse_contrastive_search(input_ids, sample_step, nucleus_p, beam_width, alpha, decoding_len)
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output))
```
More detailed tutorial on how to use contrastive search/diverse contrastive search can be found [[here]](https://github.com/yxuansu/SimCTG/tree/main/document_generation).


<span id='wikitext103_tutorial'/>

#### 3. Document Generation:
The detailed tutorial of experiment on Wikitext-103 is provided [[here]](https://github.com/yxuansu/SimCTG/tree/main/document_generation).

<span id='dialogue_tutorial'/>

#### 4. Dialogue Generation:
The detailed tutorial 
