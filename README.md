# A Contrastive Framework for Neural Text Generation
**Authors**: Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, and Nigel Collier

Code of our paper: [A Contrastive Framework for Neural Text Generation]()

****

### News:
[2022//] SimCTG is publicly released!

****
## Introduction

****
## Citation:
If you find our paper and resources useful, please kindly star this repo and cite our paper:

```bibtex
@article{SuSimCTG2022,
  author    = {Yixuan Su and
               Tian Lan and
               Yan Wang and
               Dani Yogatama and
               Lingpeng Kong and
               Nigel Collier},
  title     = {A Contrastive Framework for Neural Text Generation},
  journal   = {CoRR},
  year      = {2022},
  eprinttype = {arXiv}
}
```

****
## Huggingface Models
### 1. Wikitext-103:

|Model Name|Model Address|
|:-------------:|:-------------:|
|cambridgeltl/simctg_wikitext103 (**cambridgeltl/tacl-bert-base-uncased**)|[link](https://huggingface.co/cambridgeltl/tacl-bert-base-uncased)|
|cambridgeltl/simctg_lccc (**cambridgeltl/tacl-bert-base-chinese**)|[link](https://huggingface.co/cambridgeltl/tacl-bert-base-chinese)|
|cambridgeltl/simctg_english_wikipedia (**cambridgeltl/tacl-bert-base-chinese**)|[link](https://huggingface.co/cambridgeltl/tacl-bert-base-chinese)|

****
## Catalogue:
* <a href='#environment_setup'>1. Environment Setup</a>
* <a href='#example_usage'>2. Example Usage of Contrastive Search</a>
* <a href='#wikitext103_tutorial'>3. Document Generation</a>
* <a href='#dialogue_tutorial'>4. Open-domain Dialogue Generation</a>
* <a href='#pretraining'>5. General Domain Pre-training</a>


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
The detailed tutorial of experiment on document generation is provided [[here]](https://github.com/yxuansu/SimCTG/tree/main/document_generation).

<span id='dialogue_tutorial'/>

#### 4. Open-domain Dialogue Generation:
The detailed tutorial of experiment on open-domain dialogue generation provided [[here]](https://github.com/yxuansu/SimCTG/tree/main/dialogue_generation).


### Contact
If you have any questions, feel free to contact me via (ys484 at cam.ac.uk).
