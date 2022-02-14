# A Contrastive Framework for Neural Text Generation
**Authors**: Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, and Nigel Collier

This repository contains code, models, and other related resources of our paper [A Contrastive Framework for Neural Text Generation]().

****
### Catalogue:
* <a href='#introduction'>1. Introduction</a>
* <a href='#news'>2. News</a>
* <a href='#citation'>3. Citation</a>
* <a href='#models'>4. Huggingface Models</a>
* <a href='#environment_setup'>5. Environment Setup</a>
* <a href='#example_usage'>6. Example Usage of Contrastive Search</a>
    * <a href='#example_usage_english_simctg'>6.1. Use SimCTG Pretrained on Wikipedia Corpus</a>
    * <a href='#example_usage_different_language_model'>6.2. Use **Off-the-shelf** Language Models from Different Languages</a>
        * <a href='#example_usage_chinese_gpt'>6.2.1. Chinese Language Model</a>
        * <a href='#example_usage_japanese_gpt'>6.2.2. Japanese Language Model</a>
* <a href='#wikitext103_tutorial'>7. Document Generation</a>
* <a href='#dialogue_tutorial'>8. Open-domain Dialogue Generation</a>
* <a href='#pretraining'>9. Large-Scale Pre-training with SimCTG</a>
* <a href='#contact'>10. Contact</a>



****

<span id='introduction'/>

#### 1. Introduction:
Text generation is of great importance to many natural language processing applications. However, maximization-based decoding methods (e.g. beam search) of neural language models often lead to degenerate solutions---the generated text is unnatural and contains undesirable repetitions. Existing approaches introduce stochasticity via sampling or modify training objectives to decrease probabilities of certain tokens (e.g., unlikelihood training). However, they often lead to solutions that lack coherence. In this work, we show that an underlying reason for model degeneration is the anisotropic distribution of token representations. We present a contrastive solution: (i) SimCTG, a contrastive training objective to calibrate the model's representation space, and (ii) a decoding method---contrastive search---to encourage diversity while maintaining coherence in the generated text. Extensive experiments and analyses on three benchmarks from two languages demonstrate that our proposed approach outperforms state-of-the-art text generation methods as evaluated by both human and automatic metrics.
****

<span id='news'/>

#### 2. News:
[2022/02/14] SimCTG is publicly released!


****

<span id='citation'/>

#### 3. Citation:
If you find our paper and resources useful, please kindly leave a star and cite our paper. Thanks!

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

<span id='models'/>

#### 4. Huggingface Models:

|Model Name|Task|Language|Training Corpus (Size)|Model Size|Model Address|
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|cambridgeltl/simctg_wikitext103|Document Generation|English|Wikitext-103 (529MB)|117M|[[link]](https://huggingface.co/cambridgeltl/simctg_wikitext103/)|
|cambridgeltl/simctg_lccc_dialogue|Open-domain Dialogue Generation|Chinese|LCCC (708MB)|117M|[[link]](https://huggingface.co/cambridgeltl/simctg_lccc_dialogue/)|
|cambridgeltl/simctg_english_wikipedia|General Domain Pre-training|English|Wikipedia (14.11GB)|117M|[[link]](https://huggingface.co/cambridgeltl/simctg_english_wikipedia/)|


****

<span id='environment_setup'/>

#### 5. Environment Setup:
```yaml
python version: 3.8
pip3 install -r requirements.txt
```
****

<span id='example_usage'/>

#### 6. Example Usage of Contrastive Search:

<span id='example_usage_english_simctg'/>

##### 6.1. Use SimCTG Pretrained on Wikipedia Corpus:
Here, we show how to use contrastive search to generate the result.
```python
import torch
import sys
sys.path.append(r'./pretraining')
from simctg import SimCTGPretraining
# load SimCTG model pretrained on the large-scale Wikipedia corpus
model_path = r'cambridgeltl/simctg_english_wikipedia'
model = SimCTGPretraining(model_path)
model.eval()

# we randomly select a prefix from the dev set of Wikipedia pre-training corpus and prepare the text prefix input
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
More details on how to pre-train SimCTG on large-scale corpus and the details of the argument setup in contrastive search can be found [[here]](https://github.com/yxuansu/SimCTG/tree/main/pretraining).


<span id='example_usage_different_language_model'/>

##### 6.2. Use Off-the-shelf Language Models from Different Languages:
Importantly, we found that contrastive search can be directly applied to **off-the-shelf** language models even **without** contrastive training. The only condition is that <ins>the corresponding language should be naturally tokenized by character units</ins>. Some examples include Chinese, Japanese, and Korean. In the following, we showcase how to use contrastive search with off-the-shelf Chinese and Japanese language models. More analysis of why contrastive search works well on vanilla language models can be found in the Appendix C of our paper. 


<span id='example_usage_chinese_gpt'/>

###### 6.2.1. Chinese Language Model:
```python
import torch
import sys
sys.path.append(r'./pretraining')
from simctg import SimCTGPretraining
# load an off-the-shelf Chinese GPT (https://huggingface.co/uer/gpt2-chinese-cluecorpussmall)
model_path = r'uer/gpt2-chinese-cluecorpussmall'
model = SimCTGPretraining(model_path)
model.eval()

# prepare text prefix input
text = r'苹果公司'
tokens = model.tokenizer.tokenize(text)
input_ids = model.tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor(input_ids).view(1,-1)

# (1) use contrastive search to generate the result
beam_width, alpha, decoding_len = 3, 0.6, 128
eos_token = '[SEP]'
print (model.fast_contrastive_search(input_ids, beam_width, alpha, decoding_len, eos_token))
'''
   '苹果公司在中国市场推出的iphone7，不仅在外观设计上有所改变，在配置上也进行了升级。苹果还宣布，新一代iphone将采用
   5.7英寸屏幕，分辨率达到2560×1440像素，显示效果非常出色。此外，该机还支持指纹识别功能，可实现手指快速扫描、人脸识
   别等功能。'
'''

# (2) use nucleus sampling to generate the result
nucleus_p, decoding_len = 0.95, 128
eos_token = '[SEP]'
print (model.nucleus_sampling(input_ids, nucleus_p, decoding_len, eos_token))
'''
   '苹果公司的设计套件。2.不同的颜色设计有不同的热塑性材质。热塑性材质中的ca34bc是真正能够让人感觉舒适的材质。3.比利
   时家具建筑师埃莉诺特·夏格和大家举一些非常实用又非常普遍的例子在这里艾格的设计师们会简单介绍一下为什么美国家具是比利
   时建筑的一个分支或一个分支，他们'
'''

# (3) use greedy search to generate the result
decoding_len = 128
eos_token = '[SEP]'
print (model.greedy_search(input_ids, decoding_len, eos_token))
'''
   '苹果公司的一个重要客户，他们的产品在全球范围内都有着非常高的知名度。[UNK]我们的产品在全球范围内都有着非常高的知名度，
   我们的产品在全球范围内都有着非常高的知名度。[UNK]在这样的背景下，苹果公司的产品在全球范围内都有着非常高的知名度。[UNK]
   我们的产品在全球范围内都有着非常高的知'
'''

# (4) use beam search to generate the result
beam_width, decoding_len = 10, 128
eos_token = '[SEP]'
print (model.beam_search(input_ids, 10, decoding_len, eos_token))
'''
  '苹果公司总裁兼首席执行官蒂姆·库克（timcook）表示：[UNK]苹果公司是全球最大的智能手机制造商之一，苹果公司是全球最大的
  智能手机制造商之一，苹果公司是全球最大的智能手机制造商之一，苹果公司是全球最大的智能手机制造商之一，苹果公司是全球最大
  的智能手机制造商之一，苹果公司是全球'
'''

# ------------------------------------------ Another Example --------------------------------------------- #
# prepare text prefix input
text = r'百节年为首，春节是中华民族最隆重的传统佳节。它不仅集中体现了中华'
tokens = model.tokenizer.tokenize(text)
input_ids = model.tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor(input_ids).view(1,-1)

# (1) use contrastive search to generate the result
beam_width, alpha, decoding_len = 3, 0.6, 128
eos_token = '[SEP]'
print (model.fast_contrastive_search(input_ids, beam_width, alpha, decoding_len, eos_token))
'''
  '百节年为首，春节是中华民族最隆重的传统佳节。它不仅集中体现了中华文化精髓，也表现了人民群众生活水平的提高和对美好生活的向往。'
'''

# (2) use nucleus sampling to generate the result
nucleus_p, decoding_len = 0.95, 128
eos_token = '[SEP]'
print (model.nucleus_sampling(input_ids, nucleus_p, decoding_len, eos_token))
'''
  '百节年为首，春节是中华民族最隆重的传统佳节。它不仅集中体现了中华传统文化，更是经济、政治、文化上的一个精神机能的全面发展。
   人们在生活中不仅能够充分认识到这个民族的非物质文化遗产，而且能够在此基础上追求书面化的概念。中国历史上有许多著名的「人物」
   ，他们深深地扎根于中国历史的传统历史文化中，热爱中华文化，热爱中华文化的传承'
'''

# (3) use greedy search to generate the result
decoding_len = 128
eos_token = '[SEP]'
print (model.greedy_search(input_ids, decoding_len, eos_token))
'''
  '百节年为首，春节是中华民族最隆重的传统佳节。它不仅集中体现了中华民族的传统美德，也体现了中华民族的传统文化。[UNK]中华民族
   的传统美德，是中华民族的传统美德。[UNK]中华民族的传统美德，是中华民族的传统美德。[UNK]中华民族的传统美德，是中华民族的传
   统美德。[UNK]中华民族的传统美德，是中华民族的传统美德。[UNK]中华民族的传统美德，是中华民族的传'
'''

# (4) use beam search to generate the result
beam_width, decoding_len = 10, 128
eos_token = '[SEP]'
print (model.beam_search(input_ids, 10, decoding_len, eos_token))
'''
  '百节年为首，春节是中华民族最隆重的传统佳节。它不仅集中体现了中华民族伟大复兴的历史使命，也体现了中华民族伟大复兴的历史使命。
   中华民族伟大复兴的历史使命，不仅体现了中华民族伟大复兴的历史使命，也体现了中华民族伟大复兴的历史使命。中华民族伟大复兴的历
   史使命，不仅体现了中华民族伟大复兴的历史使命，也体现了中华民族伟大复兴的历'
'''
```
More details on how to use different decoding methods to generate the result can be found [[here]](https://github.com/yxuansu/SimCTG/tree/main/pretraining).


<span id='example_usage_japanese_gpt'/>

###### 6.2.2. Japanese Language Model:

```python
import torch
import sys
sys.path.append(r'./pretraining')
from simctg import SimCTGPretraining
# load an off-the-shelf Japanese GPT (https://huggingface.co/colorfulscoop/gpt2-small-ja)
model_path = r'colorfulscoop/gpt2-small-ja'
model = SimCTGPretraining(model_path)
model.eval()

'''
   Prepare text prefix input. The prefix is copied from a random Japanese Wikipedia 
   page here (https://ja.wikipedia.org/wiki/%E8%87%A5%E9%BE%8D%E6%A1%9C).
'''
text = r'臥龍桜（がりゅうざくら）は、岐阜県高山市一之宮町にある一本桜。龍が地'
tokens = model.tokenizer.tokenize(text)
input_ids = model.tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor(input_ids).view(1,-1)

# use contrastive search to generate the result
beam_width, alpha, decoding_len = 5, 0.6, 128
eos_token = model.tokenizer.eos_token
print (model.fast_contrastive_search(input_ids, beam_width, alpha, decoding_len, eos_token))
'''
   臥龍桜(がりゅうざくら)は、岐阜県高山市一之宮町にある一本桜。龍が地中に染みつく様子を図案化したもので、樹齢400年
   を越す日本さくら名所100選に選定されている。一之宮町指定天然記念物。岐阜県飛騨地方(東濃地方)の山間地に生育し、約
   1万年前に絶滅したと考えられている。「花の本」とも称され、開花期は5月上旬から下旬までで、桜の枝張りは濃緑色である。
   花は直径約10cmの花弁を咲かせる八重咲きで、花弁の色は紅紫色で、雄しべは4本、雌しべは1本ある。雄しべの先
'''
```

**[Note]** Sadly, I do not speak Japanese (I wish I do!), so I can only judge the quality of the generated text using [Google translate](https://translate.google.com/). It would be great if anyone could tell me whether the generated text is good or not. Thank you in advance!

****

<span id='wikitext103_tutorial'/>

#### 7. Document Generation:
The detailed tutorial of experiment on document generation is provided [[here]](https://github.com/yxuansu/SimCTG/tree/main/document_generation).

****

<span id='dialogue_tutorial'/>

#### 8. Open-domain Dialogue Generation:
The detailed tutorial of experiment on open-domain dialogue generation provided [[here]](https://github.com/yxuansu/SimCTG/tree/main/dialogue_generation).

****

<span id='pretraining'/>

#### 9. Large-Scale Pre-training with SimCTG
In addition to fine-tuning on downstream tasks (e.g. document generation and open-domain dialogue generation), we can also use a large-scale general domain corpus (i.e. Wikipedia) to pre-train a SimCTG model. [Here](https://github.com/yxuansu/SimCTG/tree/main/pretraining), we show the details of how to pre-train SimCTG using a large-scale English Wikipedia corpus.


****

<span id='contact'/>

#### 10. Contact
If you have any questions, feel free to contact me via (ys484 at cam.ac.uk).
