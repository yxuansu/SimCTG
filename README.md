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
    * <a href='#example_usage_english_simctg'>2.1. Use SimCTG Pretrained on Wikipedia Corpus</a>
    * <a href='#example_usage_chinese_gpt'>2.2. Use Off-the-shelf Chinese GPT</a>
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

<span id='example_usage_english_simctg'/>

##### 2.1. Use SimCTG Pretrained on Wikipedia Corpus:
Here, we show how to use contrastive search to generate the result.
```python
import torch
import sys
sys.path.append(r'./pretraining')
from simctg import SimCTGPretraining
# load SimCTG model pretrained on Wikipedia corpus
model_path = r'cambridgeltl/simctg_english_wikipedia'
model = SimCTGPretraining(model_path)
model.eval()

# prepare the text prefix input
text = r'Insect farming is the practice of raising and breeding insects as livestock, also referred to as minilivestock or micro stock. Insects may be farmed for the commodities'
tokens = model.tokenizer.tokenize(text)
input_ids = model.tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor(input_ids).view(1,-1)

# use contrastive search to generate the result
beam_width, alpha, decoding_len = 5, 0.6, 128
eos_token = '<|endoftext|>'
print (model.fast_contrastive_search(input_ids, beam_width, alpha, decoding_len, eos_token))

'''
   Insect farming is the practice of raising and breeding insects as livestock, also referred to as minilivestock
   or micro stock. Insects may be farmed for the  commodities they produce, such as honey, corn, sorghum, and 
   other crops. In some cases, the production of insects is a way to increase income for the owner or his family. 
   This type of farming has been described as "an economic system that benefits all people regardless of race, sex, 
   or social status" (p.\xa09). A large number of farmers in North America, Europe, and South America have used the 
   method of farming for food production in order to feed their families and livestock. The most common method of 
   farming is by hand-cropping, which consists of cutting a hole in the ground and using a saw
'''
```
More details on how to pre-train SimCTG on large-scale corpus can be found [[here]](https://github.com/yxuansu/SimCTG/tree/main/pretraining).


<span id='example_usage_chinese_gpt'/>

##### 2.2. Use Off-the-shelf Chinese GPT:
Interestingly, we found the contrastive search can work surprisingly well with Chinese GPT (**without contrastive training!**). 



<span id='wikitext103_tutorial'/>

#### 3. Document Generation:
The detailed tutorial of experiment on document generation is provided [[here]](https://github.com/yxuansu/SimCTG/tree/main/document_generation).

<span id='dialogue_tutorial'/>

#### 4. Open-domain Dialogue Generation:
The detailed tutorial of experiment on open-domain dialogue generation provided [[here]](https://github.com/yxuansu/SimCTG/tree/main/dialogue_generation).


### Contact
If you have any questions, feel free to contact me via (ys484 at cam.ac.uk).
