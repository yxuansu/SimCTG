# A Simple Contrastive Learning Framework for Neural Text Generation
**Authors**: Yixuan Su, Tian Lan, Yan Wang, Lingpeng Kong, Dani Yogatama, and Nigel Collier

Code of our paper: [A Simple Contrastive Learning Framework for Neural Text Generation]()

****
## Introduction

****
## Huggingface Models
### (1) Wikitext-103:

****
## Catalogue:
* <a href='#example_usage'>1. Example Usage</a>
* <a href='#overall_tutorial'>2. Tutorial on how to reproduce the results in our paper</a>
    * <a href='#wikitext103_tutorial'>2.1 Wikitext-103</a>

****
<span id='example_usage'/>


### 1. Example Usage:
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
More detailed tutorial on how to use contrastive search/diverse contrastive search can be found [here](https://github.com/yxuansu/SimCTG/tree/main/language_modelling).

****
<span id='overall_tutorial'/>

### 2. Tutorial on how to reproduce the results in our paper:

<span id='wikitext103_tutorial'/>
#### 2.1 Wikitext-103:
The detailed tutorial is provided [here](https://github.com/yxuansu/SimCTG/tree/main/language_modelling).
