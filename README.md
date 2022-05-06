# A Contrastive Framework for Neural Text Generation
**Authors**: Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, and Nigel Collier

This repository contains code, models, and other related resources of our paper [A Contrastive Framework for Neural Text Generation](https://arxiv.org/abs/2202.06417).

****
### Catalogue:
* <a href='#introduction'>1. Introduction</a>
* <a href='#news'>2. News</a>
* <a href='#citation'>3. Citation</a>
* <a href='#models'>4. Huggingface Models</a>
* <a href='#environment_setup'>5. Environment Setup</a>
* <a href='#example_usage'>6. Example Usage of Contrastive Search</a>
    * <a href='#example_usage_english_simctg'>6.1. Use SimCTG Pretrained on Wikipedia Corpus</a> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MhK3cVHW9HQ1ArXu0M_sS_Po0_4N1xgQ?usp=sharing)
    * <a href='#example_usage_different_language_model'>6.2. Use **Off-the-shelf** Language Models from Different Languages</a>
        * <a href='#example_usage_chinese_gpt'>6.2.1. Chinese Language Model</a> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_55LEg2caLM-lYDVIhWjxgv75IWkEry6?usp=sharing)
        * <a href='#example_usage_japanese_gpt'>6.2.2. Japanese Language Model</a> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1844kf-BttuPt1DaYhdgw-07-qz7V7pOd?usp=sharing)
        * <a href='#example_usage_korean_gpt'>6.2.3. Korean Language Model</a> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a8g1n86S-zmGe7Nb0PgVQnqWSAMfIR3D?usp=sharing)
* <a href='#wikitext103_tutorial'>7. Document Generation</a> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_zPZRlbJo5iw_Q7FUhP113udPnciUxVF?usp=sharing)
* <a href='#dialogue_tutorial'>8. Open-domain Dialogue Generation</a> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/drive/1_55LEg2caLM-lYDVIhWjxgv75IWkEry6?usp=sharing)
* <a href='#pretraining'>9. Large-Scale Pre-training with SimCTG</a>
* <a href='#story_generation'>10. Open-Ended Story Generation</a>
* <a href='#contrastive_for_encoder_decoder'>11. Contrastive Search on Encoder-Decoder Models</a>
* <a href='#contact'>12. Contact</a>
* <a href='#simctg_elsewhere'>13. SimCTG Elsewhere</a>



****

<span id='introduction'/>

#### 1. Introduction:
Text generation is of great importance to many natural language processing applications. However, maximization-based decoding methods (e.g. beam search) of neural language models often lead to degenerate solutions---the generated text is unnatural and contains undesirable repetitions. Existing approaches introduce stochasticity via sampling or modify training objectives to decrease probabilities of certain tokens (e.g., unlikelihood training). However, they often lead to solutions that lack coherence. In this work, we show that an underlying reason for model degeneration is the anisotropic distribution of token representations. We present a contrastive solution: (i) SimCTG, a contrastive training objective to calibrate the model's representation space, and (ii) a decoding method---contrastive search---to encourage diversity while maintaining coherence in the generated text. Extensive experiments and analyses on three benchmarks from two languages demonstrate that our proposed approach outperforms state-of-the-art text generation methods as evaluated by both human and automatic metrics.
****

<span id='news'/>

#### 2. News:
* [2022/05/06] :star: We have released **_MAGIC_**, a follow up work of SimCTG, that is the SOTA method in zero-shot multi-modal text generation tasks (e.g., zero-shot image captioning and visually grounded story generation). Check it out! [[paper]](https://arxiv.org/abs/2205.02655) [[code]](https://github.com/yxuansu/MAGIC)
* [2022/04/16] We have updated instructions on how to apply contrastive search on encoder-decoder models (e.g. BART and T5). More details can be found [[here]](https://github.com/yxuansu/SimCTG/tree/main/SimCTGEncDec).
* [2022/04/07] SimCTG has been publicly deployed in the **AI sentence completion** module of [Tencent's Effidit platform (腾讯智能创作助手)](https://effidit.qq.com/). Check it out and have fun!
* [2022/04/02] Add support on another benchmark (ROCStories) for the task of open-ended story generation.
* [2022/03/06] Example of how to adapt our approach to open-ended story generation is released.
* [2022/02/15] SimCTG is publicly released!


****

<span id='citation'/>

#### 3. Citation:
If you find our paper and resources useful, please kindly leave a star and cite our paper. Thanks!

```bibtex
@article{DBLP:journals/corr/abs-2202-06417,
  author    = {Yixuan Su and
               Tian Lan and
               Yan Wang and
               Dani Yogatama and
               Lingpeng Kong and
               Nigel Collier},
  title     = {A Contrastive Framework for Neural Text Generation},
  journal   = {CoRR},
  volume    = {abs/2202.06417},
  year      = {2022},
  url       = {https://arxiv.org/abs/2202.06417},
  eprinttype = {arXiv},
  eprint    = {2202.06417},
  timestamp = {Fri, 18 Feb 2022 12:23:53 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2202-06417.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
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
|cambridgeltl/simctg_writingprompts|Open-Ended Story Generation|English|WritingPrompts (865MB)|117M|[[link]](https://huggingface.co/cambridgeltl/simctg_writingprompts/)|
|cambridgeltl/simctg_rocstories|Open-Ended Story Generation|English|ROCStories (12MB)|117M|[[link]](https://huggingface.co/cambridgeltl/simctg_rocstories/)|


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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MhK3cVHW9HQ1ArXu0M_sS_Po0_4N1xgQ?usp=sharing)

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
Importantly, we found that contrastive search can be directly applied to **off-the-shelf** language models even **without** contrastive training. The only condition is that <ins>the corresponding language should be naturally tokenized by character units</ins>. Some examples include Chinese, Japanese, and Korean. In the following, we showcase how to use contrastive search with off-the-shelf Chinese, Japanese, and Korean language models. More analysis of why contrastive search works well on vanilla language models can be found in the Appendix C of our paper. 


<span id='example_usage_chinese_gpt'/>

###### 6.2.1. Chinese Language Model:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_55LEg2caLM-lYDVIhWjxgv75IWkEry6?usp=sharing)

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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1844kf-BttuPt1DaYhdgw-07-qz7V7pOd?usp=sharing)

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

# (1) use contrastive search to generate the result
beam_width, alpha, decoding_len = 5, 0.6, 128
eos_token = model.tokenizer.eos_token
print (model.fast_contrastive_search(input_ids, beam_width, alpha, decoding_len, eos_token))
'''
   臥龍桜(がりゅうざくら)は、岐阜県高山市一之宮町にある一本桜。龍が地中に染みつく様子を図案化したもので、樹齢400年
   を越す日本さくら名所100選に選定されている。一之宮町指定天然記念物。岐阜県飛騨地方(東濃地方)の山間地に生育し、約
   1万年前に絶滅したと考えられている。「花の本」とも称され、開花期は5月上旬から下旬までで、桜の枝張りは濃緑色である。
   花は直径約10cmの花弁を咲かせる八重咲きで、花弁の色は紅紫色で、雄しべは4本、雌しべは1本ある。雄しべの先
'''

# (2) use nucleus sampling to generate the result
nucleus_p, decoding_len = 0.95, 128
eos_token = model.tokenizer.eos_token
print (model.nucleus_sampling(input_ids, nucleus_p, decoding_len, eos_token))
'''
   臥龍桜(がりゅうざくら)は、岐阜県高山市一之宮町にある一本桜。龍が地中に棲む奇岩に由来する。毎年5月上旬には多くの花見
   客が訪れている。かつて、雪見の藩お抱え家臣、雲口である長久城主長久竜泰が祭っている「月輪寺」には手水鉢が2つあり、長
   久氏の勢力が強まると同時に関連する寺もあり、山を挟むように吉野側の赤峰山から北へ順に樹齢250年を越してきたが、江戸時
   代に廃材が搬出されてから薪が取れなくなっている。古い株は毎年12月の初午に燃えつき風雨が吹き荒れて朽ち果てる。根は分枝
'''

# (3) use greedy search to generate the result
decoding_len = 128
eos_token = model.tokenizer.eos_token
print (model.greedy_search(input_ids, decoding_len, eos_token))
'''
   臥龍桜(がりゅうざくら)は、岐阜県高山市一之宮町にある一本桜。龍が地中に棲む龍の棲むとされる桜で、樹齢は1000年以上。樹
   高は10mほどで、幹周りは8mほどになる。樹齢は300年ほどで、樹高は20mほどになる。樹形が整っており、枝張りも良く、樹勢も
   旺盛である。樹形は、樹高が1mほどで、幹周りは4mほどになる。枝張りはよく発達し、樹勢は旺盛である。冬になると、幹周りの
   樹冠が紅葉する。また、紅葉の時期には、樹冠が赤く紅葉する。樹
'''

# (4) use beam search to generate the result
beam_width, decoding_len = 10, 128
eos_token = model.tokenizer.eos_token
print (model.beam_search(input_ids, 10, decoding_len, eos_token))
'''
   臥龍桜(がりゅうざくら)は、岐阜県高山市一之宮町にある一本桜。龍が地中深くに咲く桜で、岐阜県の天然記念物に指定されている。
   岐阜県高山市一之宮町一之宮(いちのみやちょういちのみや)は、岐阜県高山市一之宮町一之宮にある一本桜である。龍が地中深くに
   咲く桜で、岐阜県の天然記念物に指定されている。岐阜県高山市一之宮町一之宮(いちのみやちょういちのみや)は、岐阜県高山市一
   之宮町一之宮(いちのみやちょういちのみや)は、岐阜県高山市一之宮町一之宮(いちのみやちょういちのみや)は、岐阜県高山
'''
```

**[Note]** Sadly, I do not speak Japanese (I wish I do!), so I can only judge the quality of the generated text using [Google translate](https://translate.google.com/). It would be great if anyone could tell me whether the generated text is good or not. Thank you in advance!

****

<span id='example_usage_korean_gpt'/>

###### 6.2.3. Korean Language Model:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a8g1n86S-zmGe7Nb0PgVQnqWSAMfIR3D?usp=sharing)

```python
import torch
import sys
sys.path.append(r'./pretraining')
from simctg import SimCTGPretraining
# load an off-the-shelf Korean GPT (https://huggingface.co/skt/ko-gpt-trinity-1.2B-v0.5)
model_path = r'skt/ko-gpt-trinity-1.2B-v0.5'
model = SimCTGPretraining(model_path)
model.eval()

'''
   Prepare text prefix input.
'''
text = r'인간처럼 생각하고, 행동하는 \'지능\'을 통해 인류가 이제까지 풀지 못했던'
tokens = model.tokenizer.tokenize(text)
input_ids = model.tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor(input_ids).view(1,-1)

# (1) use contrastive search to generate the result
beam_width, alpha, decoding_len = 5, 0.6, 64 
# because this model is pretty large, so we set the generation length (decoding_len) as 64
eos_token = model.tokenizer.eos_token
print (model.fast_contrastive_search(input_ids, beam_width, alpha, decoding_len, eos_token))
'''
   인간처럼생각하고,행동하는\'지능\'을통해인류가이제까지풀지못했던난제를해결하려한다.이책의제목이기도한'슈퍼인텔리전스'는인공지능
   (AI)의등장으로야기된사회변화를일컫는말로,이책을관통하는키워드이기도하다.저자는"기술과인간사이의경계가무너지고있다"고지적한다.
   AI가인간의사고방식과행동을모방할뿐만
'''

# (2) use nucleus sampling to generate the result
nucleus_p, decoding_len = 0.95, 64
eos_token = model.tokenizer.eos_token
print (model.nucleus_sampling(input_ids, nucleus_p, decoding_len, eos_token))
'''
  '인간처럼생각하고,행동하는\'지능\'을통해인류가이제까지풀지못했던큰수수께끼를풀수있다.'지능\'은인공두뇌그자체이기도하지만그공간의
  반영이라는해석도가능하다.예를들면시간부등호처럼복잡한수식을쉽게떠올릴수있다는이야기다.마치구글에검색창에'Quick'이라는단어를입력하
  면자동으로'중력'은일정한법칙에따라'
'''

# (3) use greedy search to generate the result
decoding_len = 64
eos_token = model.tokenizer.eos_token
print (model.greedy_search(input_ids, decoding_len, eos_token))
'''
  '인간처럼생각하고,행동하는\'지능\'을통해인류가이제까지풀지못했던문제를해결할수있다고주장한다.이지능은\'지능\'그자체라기보다\'지능\'
  그자체를구성하는\'지능\'그자체라고할수있다.이지능은\'지능\'그자체라기보다\'지능\'그자체를구성하는\'지능\'그자체라고'
'''

# (4) use beam search to generate the result
# We do not print the result, because beam search stops generation immediately.
```

**[Note]** Sadly, I am not a Korean speaker either, so I can only judge the quality of the generated text using [Google translate](https://translate.google.com/) as well. It would be great if anyone could tell me whether the generated text is good or not. Thank you!

****

<span id='wikitext103_tutorial'/>

#### 7. Document Generation:
The detailed tutorial of experiment on document generation is provided [[here]](https://github.com/yxuansu/SimCTG/tree/main/document_generation).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_zPZRlbJo5iw_Q7FUhP113udPnciUxVF?usp=sharing)

****

<span id='dialogue_tutorial'/>

#### 8. Open-domain Dialogue Generation:
The detailed tutorial of experiment on open-domain dialogue generation provided [[here]](https://github.com/yxuansu/SimCTG/tree/main/dialogue_generation).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/drive/1_55LEg2caLM-lYDVIhWjxgv75IWkEry6?usp=sharing)

****

<span id='pretraining'/>

#### 9. Large-Scale Pre-training with SimCTG
In addition to fine-tuning on downstream tasks (e.g. document generation and open-domain dialogue generation), we can also use a large-scale general domain corpus (i.e. Wikipedia) to pre-train a SimCTG model. [Here](https://github.com/yxuansu/SimCTG/tree/main/pretraining), we show the details of how to pre-train SimCTG using a large-scale English Wikipedia corpus.


****

<span id='story_generation'/>

#### 10. Open-Ended Story Generation
We also show how to adapt our approach to open-ended story generation task. The details are provided [[here]](https://github.com/yxuansu/SimCTG/tree/main/story_generation).


****

<span id='contrastive_for_encoder_decoder'/>

#### 11. Contrastive Search on Encoder-Decoder Models

Details on how to apply contrastive search on encoder-decoder models (e.g. BART and T5) can be found [[here]](https://github.com/yxuansu/SimCTG/tree/main/SimCTGEncDec).

****

<span id='contact'/>

#### 12. Contact
If you have any questions, feel free to contact me via (ys484 at cam.ac.uk).

****

<span id='simctg_elsewhere'/>

#### 13. SimCTG Elsewhere

We thank the community's effort for extending SimCTG!

- [Zenn](https://zenn.dev/) has provided [a tutorial and implementation of contrastive search based on T5](https://zenn.dev/kwashizzz/articles/ml-simctg-contrastive-framework). 
- [Tencent's Effidit platform (腾讯智能创作助手)](https://effidit.qq.com/) has integrated SimCTG into its AI sentence completion module. Check it out and have fun! 

