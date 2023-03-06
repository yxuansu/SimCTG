# A Contrastive Framework for Neural Text Generation
**Authors**: Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, and Nigel Collier

This repository contains code, models, and other related resources of our paper ["A Contrastive Framework for Neural Text Generation"](https://arxiv.org/abs/2202.06417).

:star2: Check out this great [[blog]](https://huggingface.co/blog/introducing-csearch) as well as this awesome [[demo]](https://huggingface.co/spaces/joaogante/contrastive_search_generation) that are generously supported by Huggingface ([@huggingface](https://github.com/huggingface) :hugs:) which compares contrastive search with other popular decoding methods. Many thanks to Huggingface :hugs:! 

**[Use contrastive search in Huggingface transformers]** In this <a href='#contrastive_in_transformers'>tutorial</a>, we demonstrate how to use contrastive search in Huggingface `transformers`.


****
If you find our paper and resources useful, please kindly leave a star and cite our papers. Thanks!

```bibtex
@inproceedings{su2022a,
  title={A Contrastive Framework for Neural Text Generation},
  author={Yixuan Su and Tian Lan and Yan Wang and Dani Yogatama and Lingpeng Kong and Nigel Collier},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year={2022},
  url={https://openreview.net/forum?id=V88BafmH9Pj}
}

@article{su2023contrastive,
  title={Contrastive Search Is What You Need For Neural Text Generation},
  author={Yixuan Su and Nigel Collier},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2023},
  url={https://openreview.net/forum?id=GbkWw3jwL9}
}
```

****

### News:
* [2022/11/22] Released a technical report that compares Contrastive Search with another recently proposed method, i.e. Contrastive Decoding. Check out our [[paper]](https://arxiv.org/abs/2211.10797) and [[code]](https://github.com/yxuansu/Contrastive_Search_versus_Contrastive_Decoding).
* [2022/11/08] :fire: Contrastive Search has now officially supported by HuggingFace for both PyTorch and TensorFlow platforms! Check out this great [[blog]](https://huggingface.co/blog/introducing-csearch) as well as this awesome [[demo]](https://huggingface.co/spaces/joaogante/contrastive_search_generation) that are generously supported by Huggingface ([@huggingface](https://github.com/huggingface) :hugs:).
* [2022/10/26] :fire: We have released a new manuscript ["Contrastive Search Is What You Need For Neural Text Generation"](https://arxiv.org/abs/2210.14140) which has two takeaways: (1) Autoregressive language models are naturally isotropic, therefore *SimCTG training may not be necessary*; (2) Contrastive search works exceptionally well on **off-the-shelf** language models across 16 languages. On 12 out of the 16 evaluated languages, it even performs comparably with human-written text! [Paper](https://arxiv.org/abs/2210.14140) and [code](https://github.com/yxuansu/Contrastive_Search_Is_What_You_Need) are all released. Check it out!
* [2022/10/13] We have added a concise explanation on the implementations of contrastive search. Please find it [[here]](https://github.com/yxuansu/SimCTG/tree/main/contrastive_search_explanation).
* [2022/09/14] :blush: SimCTG is accepted to NeurIPS 2022!
* [2022/09/06] :fire: We have added supports for the newly released OPT models (see ["OPT: Open Pre-trained Transformer Language Models"](https://arxiv.org/abs/2205.01068)) by Meta. To see how to apply contrastive search on OPT models, check it [[here]](#contrastive_search_with_opt)!
* [2022/06/03] :fire: We have released an easy-to-use library (i.e., simctg) which allows you to use SimCTG with a simple **pip install simctg** and **a few lines of code**. Check the comprehensive and huggingface-style tutorials <a href='#tutorial'>[here]</a> and [[here]](https://github.com/yxuansu/SimCTG/tree/main/simctg)!
* [2022/05/06] :star: We have released **_MAGIC_**, a follow up work of SimCTG, that is the SOTA method in zero-shot multi-modal text generation tasks (e.g., zero-shot image captioning and visually grounded story generation). Check it out! [[paper]](https://arxiv.org/abs/2205.02655) [[code]](https://github.com/yxuansu/MAGIC)
* [2022/04/16] We have updated instructions on how to apply contrastive search on encoder-decoder models (e.g. BART and T5). More details can be found [[here]](https://github.com/yxuansu/SimCTG/tree/main/SimCTGEncDec).
* [2022/04/07] SimCTG has been publicly deployed in the **AI sentence completion** module of [Tencent's Effidit platform (腾讯智能创作助手)](https://effidit.qq.com/). Check it out and have fun!
* [2022/04/02] Add support on another benchmark (ROCStories) for the task of open-ended story generation.
* [2022/03/06] Example of how to adapt our approach to open-ended story generation is released.
* [2022/02/15] SimCTG is publicly released!

****

<span id='all_catelogue'/>

### Catalogue:
* <a href='#contrastive_in_transformers'>Apply Contrastive Search in Huggingface Transformers</a>
* <a href='#introduction'>1. Introduction</a>
* <a href='#contrastive_search_with_LMs'>2. Contrastive Search with GPT-2 and OPT :fire:</a>
    * <a href='#install_simctg_package'>2.1. Environment Setup</a>
    * <a href='#contrastive_search_with_gpt2'>2.2. Contrastive Search with GPT-2</a>
    * <a href='#contrastive_search_with_opt'>2.3. Contrastive Search with OPT</a>
* <a href='#models'>3. Huggingface Models</a>
* <a href='#tutorial'>4. Huggingface-Style Tutorials</a> :star: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ImvR-ldHf9rJyFzOCMJ_zjAGK8n1iTW7?usp=sharing)
    * <a href='#install_simctg'>4.1. Install and Load SimCTG</a>
    * <a href='#example_train_with_simctg'>4.2. Example of Training Language Model with SimCTG</a>
        * <a href='#init_simctg'>4.2.1. Initialize Language Model</a>
        * <a href='#init_loss_class'>4.2.2. Initialize Loss Class</a>
        * <a href='#init_training_data'>4.2.3. Create Example Training Data</a>
        * <a href='#compute_loss'>4.2.4. Compute Loss</a>
    * <a href='#contrastive_search_examples'>4.3. Examples of Performing Generation with Contrastive Search</a>
        * <a href='#example_document_generation'>4.3.1. Open-Ended Document Generation</a>
        * <a href='#example_dialogue_generation'>4.3.2. Open-Domain Dialogue Generation</a>
    * <a href='#example_off_the_shelf_generation'>4.4. Contrastive Search with Off-the-shelf Language Models from Different Languages</a>
        * <a href='#chinese_example_off_the_shelf_generation'>4.4.1. Chinese Language Model</a>
        * <a href='#japanese_example_off_the_shelf_generation'>4.4.2. Japanese Language Model</a>
        * <a href='#korean_example_off_the_shelf_generation'>4.4.3. Korean Language Model</a>
    * <a href='#training_tutorial'>4.5. Detailed Tutorial of Training SimCTG on Wikitext-103</a> :star:
    * <a href='#T5_tutorial'>4.6. Apply SimCTG on T5</a> :star:
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

<span id='contrastive_in_transformers'/>

#### Apply Contrastive Search in Huggingface Transformers:
Here, we demonstrate how to use contrastive search in Huggingface `transformers`.

##### (1) Install Environment:
First, to install the required packages, please run the following commands:
```yaml
pip install torch
pip install "transformers>=4.24.0"
```

##### (2) Generate Text with Contrastive Search:
Next, we show how to reproduce the result as in <a href='#contrastive_search_with_gpt2'>Section 2.2</a>

```python
# load the LMs
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model_name = 'gpt2-large'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# prepare the prefix
prefix_text = r"DeepMind Company is"
input_ids = tokenizer(prefix_text, return_tensors='pt').input_ids

# generate the result with contrastive search
output = model.generate(input_ids, penalty_alpha=0.6, top_k=4, max_length=512)
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("" + 100 * '-')
```

<details>
<summary><b>Model Output: [click to expand]</b></summary>
  
```
Output:
----------------------------------------------------------------------------------------------------  
DeepMind Company is a leader in artificial intelligence (AI). We have a long history of working
with companies such as Google, Facebook, Amazon, and Microsoft to build products that improve
people's lives, and today we are excited to announce that DeepMind's AlphaGo program has won the
game of Go, becoming the first program to defeat a professional Go player.

The victory is a testament to the power of deep learning, and to the incredible work of our
research team, which has been at the forefront of AI research for the past five years. AlphaGo
is one of the most advanced Go programs ever created, and its performance is an important step
towards the goal of human-level AI.

"This is the culmination of a decade of hard work," said Andy Ng, co-founder and CTO of DeepMind.
"We are thrilled to have achieved this milestone and look forward to continuing to develop AI that
can be used in a wide range of applications and to help people live better lives."

DeepMind's work on Go began in 2010, when it began to train a neural network to play Go using
millions of games played by top Go players around the world. Since then, the team has refined the
algorithm, adding more and more layers of reinforcement learning to make it better at recognizing
patterns and making decisions based on those patterns. In the past year and a half, the team has
made significant progress in the game, winning a record-tying 13 games in a row to move into the
top four of the world rankings.

"The game of Go is a complex game in which players have to be very careful not to overextend their
territory, and this is something that we have been able to improve over and over again," said
Dr. Demis Hassabis, co-founder and Chief Scientific Officer of DeepMind. "We are very proud of our
team's work, and we hope that it will inspire others to take the next step in their research and
apply the same techniques to other problems."

In addition to the win in Go, DeepMind has also developed an AI system that can learn to play a
number of different games, including poker, Go, and chess. This AI system, called Tarsier, was
developed in partnership with Carnegie Mellon University and the University of California, 
Berkeley, and is being used to teach computer vision and machine learning to identify objects in
images and recognize speech in natural language. Tarsier has been trained to play the game of Go
and other games on a number of different platforms...
----------------------------------------------------------------------------------------------------
```
</details>

##### (3) Huggingface Demo:

Also check out this awesome [[demo]](https://huggingface.co/spaces/joaogante/contrastive_search_generation) generously supported by Huggingface ([@huggingface](https://github.com/huggingface) :hugs:) which compares contrastive search with other popular decoding methods. Many thanks to Huggingface!

****

<span id='introduction'/>

#### 1. Introduction: <a href='#all_catelogue'>[Back to Top]</a>
Text generation is of great importance to many natural language processing applications. However, maximization-based decoding methods (e.g. beam search) of neural language models often lead to degenerate solutions---the generated text is unnatural and contains undesirable repetitions. Existing approaches introduce stochasticity via sampling or modify training objectives to decrease probabilities of certain tokens (e.g., unlikelihood training). However, they often lead to solutions that lack coherence. In this work, we show that an underlying reason for model degeneration is the anisotropic distribution of token representations. We present a contrastive solution: (i) SimCTG, a contrastive training objective to calibrate the model's representation space, and (ii) a decoding method---contrastive search---to encourage diversity while maintaining coherence in the generated text. Extensive experiments and analyses on three benchmarks from two languages demonstrate that our proposed approach outperforms state-of-the-art text generation methods as evaluated by both human and automatic metrics.
****


<span id='contrastive_search_with_LMs'/>

#### 2. Contrastive Search with GPT-2 and OPT: <a href='#all_catelogue'>[Back to Top]</a>
In this section, we illustrate how to apply contrastive search on GPT-2 models and the [OPT](https://arxiv.org/abs/2205.01068) models released by Meta.

<span id='install_simctg_package'/>

##### 2.1. Environment Setup:
To install our simctg package, simply using the command below. More details of the simctg package can be found <a href='#tutorial'>[here]</a> and [[here]](https://github.com/yxuansu/SimCTG/tree/main/simctg).
```yaml
pip install simctg --upgrade
```

<span id='contrastive_search_with_gpt2'/>

##### 2.2. Contrastive Search with GPT-2:
Let's see how to produce text with contrastive search using GPT-2 models. More details can be found [here]((https://github.com/yxuansu/SimCTG/tree/main/simctg)).

(i) First, we load the GPT-2 model as
```python
import torch
from simctg.simctggpt import SimCTGGPT
model_name = r'gpt2-large'
model = SimCTGGPT(model_name)
model.eval()
tokenizer = model.tokenizer
eos_token_id = tokenizer.eos_token_id
```

(ii) Then, we prepare the prefix text as
```python
prefix_text = r"DeepMind Company is"
tokens = tokenizer.tokenize(prefix_text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor(input_ids).view(1,-1)
```

(iii) Last, we generate the text with contrastive search as
```python
beam_width, alpha, decoding_len = 4, 0.6, 256
output = model.fast_contrastive_search(input_ids=input_ids, beam_width=beam_width, 
                                       alpha=alpha, decoding_len=decoding_len,
                                      end_of_sequence_token_id = eos_token_id, early_stop = True) 
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output))
print("" + 100 * '-')
```
<details open>
<summary><b>Model Output:</b></summary>
  
```
Output:
----------------------------------------------------------------------------------------------------
DeepMind Company is a leader in artificial intelligence (AI). We have a long history of working with 
companies such as Google, Facebook, Amazon, and Microsoft to build products that improve people's 
lives, and today we are excited to announce that DeepMind's AlphaGo program has won the game of Go,
becoming the first program to defeat a professional Go player.

The victory is a testament to the power of deep learning, and to the incredible work of our research
team, which has been at the forefront of AI research for the past five years. AlphaGo is one of the
most advanced Go programs ever created, and its performance is an important step towards the goal of
human-level AI.

"This is the culmination of a decade of hard work," said Andy Ng, co-founder and CTO of DeepMind. 
"We are thrilled to have achieved this milestone and look forward to continuing to develop AI that can
be used in a wide range of applications and to help people live better lives."

DeepMind's work on Go began in 2010, when it began to train a neural network to play Go using millions
of games played by top Go players around the world. Since then, the team has refined the algorithm,
adding more and more layers of reinforcement learning to make it...
----------------------------------------------------------------------------------------------------
```
</details>

For comparison, let's see the result generated by **greedy search**.
```python
decoding_len = 256
output = model.greedy_search(input_ids=input_ids, decoding_len=decoding_len,
                                       end_of_sequence_token_id = eos_token_id, early_stop = True) 
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output))
print("" + 100 * '-')
```

<details>
<summary><b>Model Output:</b></summary>
  
```
Output:
----------------------------------------------------------------------------------------------------
DeepMind Company is a leading AI research company, with a focus on deep learning and deep learning-based systems.

The company's research is focused on the development of deep learning-based systems that can learn from large 
amounts of data, and that can be used to solve real-world problems.

DeepMind's research is also used by the UK government to develop new technologies for the UK's National Health Service.

DeepMind's research is also used by the UK government to develop new technologies for the UK's National Health Service.

DeepMind's research is also used by the UK government to develop new technologies for the UK's National Health Service.

DeepMind's research is also used by the UK government to develop new technologies for the UK's National Health Service.

DeepMind's research is also used by the UK government to develop new technologies for the UK's National Health Service.

DeepMind's research is also used by the UK government to develop new technologies for the UK's National Health Service.

DeepMind's research is also used by the UK government to develop new technologies for the UK's National Health Service.

DeepMind's research is also used by the UK government to develop new technologies for the UK's National Health Service...
----------------------------------------------------------------------------------------------------
```
</details>

We can also see the result generated by **nucleus sampling** (p=0.95).
```python
decoding_len = 256
output = model.nucleus_sampling(input_ids=input_ids, decoding_len=decoding_len, nucleus_p=0.95,
                                       end_of_sequence_token_id = eos_token_id, early_stop = True) 
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output[1:]))
print("" + 100 * '-')
```

<details>
<summary><b>Model Output:</b></summary>
  
```
Output:
----------------------------------------------------------------------------------------------------
DeepMind Company is a Cardiff-based start-up with an exclusive mission to build the world’s largest ever
deep-learning system to analyse the world’s digital content and in particular, super-sized image content.
  
The system, the largest in the world with no previous expertise in image or digital content detection, 
will have previously relied on a mixture of machine learning, artificial neural networks, and storage,
processing and retrieval techniques.
  
The AI system, called ImageNet, will take new approach to our challenge of data science and machine
learning, significantly improving efficiency, natural language processing and full understanding of
complex, high-dimensional images, with an Eye of the Tiger framework for extracting techniques to
ensure correct detection of particular images in complex scenes.
  
Dr. Mark Ward, Dr. Alex Kudle, Dr. Ralph Pinchbeck and CTO, DeepMind Dr. Alex Kudle
  
Case Study: Derpy’s Most Wanted: Fighting Cybersecurity, building a robot-aided smuggling network
  
InfoSec News, 06/07/2017
  
Dimitrios Papadimitriou (left) and Chris Bardy (right) at G+ XE, July 2017
  
How to model an industrial malware botn...
----------------------------------------------------------------------------------------------------
```
</details>


<span id='contrastive_search_with_opt'/>

##### 2.3. Contrastive Search with OPT:
Let's see how to produce text with contrastive search using OPT models. More details can be found [here]((https://github.com/yxuansu/SimCTG/tree/main/simctg)).

(i) First, we load the OPT model as
```python
import torch
from simctg.simctgopt import SimCTGOPT
model_name = 'facebook/opt-6.7b'
model = SimCTGOPT(model_name)
tokenizer = model.tokenizer
model.eval()
bos_token_id = tokenizer.bos_token_id
eos_token_id = tokenizer.eos_token_id
```

(ii) Then, we use the same example from the original [paper](https://arxiv.org/abs/2205.01068) (see Figure 9 in the Appendix E) to show how to generate text with contrastive search. The prefix text is provided as
```python
prefix_text = r"""A chat between a curious human and the Statue of Liberty.

Human: What is your name?
Statue: I am the Statue of Liberty.
Human: Where do you live?
Statue: New York City.
Human: How long have you lived there?"""
```

(iii) We prepare the input ids as

**[Important Tip]** As the authors suggested in their [[tutorial]](https://huggingface.co/docs/transformers/model_doc/opt), in contrastive to GPT2, OPT adds the EOS token </s> to the beginning of every prompt. So make sure the special token is added at the front of the prompt.

```python
tokens = tokenizer.tokenize(prefix_text)
input_ids = [bos_token_id] + tokenizer.convert_tokens_to_ids(tokens) # adds </s> to the beginning of every prompt
input_ids = torch.LongTensor(input_ids).view(1,-1)
```

(iv) Last, we generate the text with contrastive search as
```python
beam_width, alpha, decoding_len = 5, 0.6, 256
output = model.fast_contrastive_search(input_ids=input_ids, beam_width=beam_width, 
                                       alpha=alpha, decoding_len=decoding_len,
                                       end_of_sequence_token_id = eos_token_id, early_stop = True) 
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output[1:]))
print("" + 100 * '-')
```

<details open>
<summary><b>Model Output:</b></summary> 

```
Output:
----------------------------------------------------------------------------------------------------
A chat between a curious human and the Statue of Liberty.

Human: What is your name?
Statue: I am the Statue of Liberty.
Human: Where do you live?
Statue: New York City.
Human: How long have you lived there?
Statue: Since 1884.
Human: Why did you come to America?
Statue: I was given to the United States by France as a gift for helping the French during the Franco-Prussian War.
Human: What do you think of America?
Statue: I love it. It is the greatest country in the world.
Human: What’s the weather like in New York?
Statue: It is cold.
Human: Is it safe to walk around at night?
Statue: Yes. There are policemen everywhere.
Human: Do you have any children?
Statue: Not yet. My pedestal is empty.
Human: What would you like to say to people who want to immigrate to America?
Statue: Come on over. You will be happy here. We have everything you need.
----------------------------------------------------------------------------------------------------
```
</details>

For comparison, let's see the result generated by **greedy search**.
```python
decoding_len = 256
output = model.greedy_search(input_ids=input_ids, decoding_len=decoding_len,
                                       end_of_sequence_token_id = eos_token_id, early_stop = True) 
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output[1:]))
print("" + 100 * '-')
```

<details>
<summary><b>Model Output:</b></summary>
  
```
Output:
----------------------------------------------------------------------------------------------------
A chat between a curious human and the Statue of Liberty.

Human: What is your name?
Statue: I am the Statue of Liberty.
Human: Where do you live?
Statue: New York City.
Human: How long have you lived there?
Statue: I have lived here for over 100 years.
Human: What do you do?
Statue: I welcome people from all over the world to come to America.
Human: What do you think of America?
Statue: I love America.
Human: What do you think of immigrants?
Statue: I love immigrants.
Human: What do you think of America?
Statue: I love America.
Human: What do you think of immigrants?
Statue: I love immigrants.
Human: What do you think of America?
Statue: I love America.
Human: What do you think of immigrants?
Statue: I love immigrants.
Human: What do you think of America?
Statue: I love America.
Human: What do you think of immigrants?
Statue: I love immigrants.
Human: What do you think of America?
Statue: I love America.
Human: What do you think of immigrants?
Statue: I love immigrants.
Human: What do you think of America?
Statue: I love America.
Human: What do you think of immigrants?
Statue: I love immigrants.
Human...
----------------------------------------------------------------------------------------------------
```
</details>

We can also see the result generated by **nucleus sampling** (p=0.95).
```python
decoding_len = 256
output = model.nucleus_sampling(input_ids=input_ids, decoding_len=decoding_len, nucleus_p=0.95,
                                       end_of_sequence_token_id = eos_token_id, early_stop = True) 
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output[1:]))
print("" + 100 * '-')
```

<details>
<summary><b>Model Output:</b></summary>
  
```
Output:
----------------------------------------------------------------------------------------------------
A chat between a curious human and the Statue of Liberty.

Human: What is your name?
Statue: I am the Statue of Liberty.
Human: Where do you live?
Statue: New York City.
Human: How long have you lived there?
Statue: Since 1876.
Human: Why is the Statue of Liberty guarded?
Statue: Because there are many people trying to steal her.

a comparison about an unexpressed thought

I would also share the story of “A Humble Fear.” At a conference in New York the Dalai Lama gave a 
speech to the International Thinkers Congress in New York. The whole thing was recorded, and the 
video is quite interesting. (on a side note, I love the fact that there were some people who laughed
when he described himself as a humble being… I think the video is hilarious, there is a reason why
I put up the video. Because if you cannot find the humor in this you’re sadly lacking…)

In the speech, the Dalai Lama compares the search for truth to searching for treasure. He says: 
“However there is a huge difference between being a thief and a collector. A thief simply takes things, 
whereas a collector looks for the beauty, even if it is just a single object.”

The above quote is perhaps the most cliched Buddhist philosophy of our times. However the comparison
between a collector and a thief is quite interesting. I like to think that the Buddha...
----------------------------------------------------------------------------------------------------
```
</details>



****

<span id='models'/>

#### 3. Huggingface Models: <a href='#all_catelogue'>[Back to Top]</a>

|Model Name|Task|Language|Training Corpus (Size)|Model Size|Model Address|
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|cambridgeltl/simctg_wikitext103|Document Generation|English|Wikitext-103 (529MB)|117M|[[link]](https://huggingface.co/cambridgeltl/simctg_wikitext103/)|
|cambridgeltl/simctg_lccc_dialogue|Open-domain Dialogue Generation|Chinese|LCCC (708MB)|117M|[[link]](https://huggingface.co/cambridgeltl/simctg_lccc_dialogue/)|
|cambridgeltl/simctg_english_wikipedia|General Domain Pre-training|English|Wikipedia (14.11GB)|117M|[[link]](https://huggingface.co/cambridgeltl/simctg_english_wikipedia/)|
|cambridgeltl/simctg_writingprompts|Open-Ended Story Generation|English|WritingPrompts (865MB)|117M|[[link]](https://huggingface.co/cambridgeltl/simctg_writingprompts/)|
|cambridgeltl/simctg_rocstories|Open-Ended Story Generation|English|ROCStories (12MB)|117M|[[link]](https://huggingface.co/cambridgeltl/simctg_rocstories/)|


****

<span id='tutorial'/>

#### 4. Huggingface-Style Tutorials: <a href='#all_catelogue'>[Back to Top]</a>
We have encapsulated our work as an easy-to-use library (i.e., package). In the following, we provide huggingface-style tutorials on how to use SimCTG and contrastive search with just **a few lines of code**! 

:star: **[Documentation]** We have provided detailed documentation of the (i) **source code** of the package and (ii) **instructions** on how to use it. Please refer to [[here]](https://github.com/yxuansu/SimCTG/tree/main/simctg).

:star: **[Google Colab]** We provide a Google Colab for the easy reproductivity of our tutorial. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ImvR-ldHf9rJyFzOCMJ_zjAGK8n1iTW7?usp=sharing)

<span id='install_simctg'/>

##### 4.1. Install and Load SimCTG:
To use our package, we recommand you to use Python with version >= 3.6. The SimCTG can be installed and loaded with the commands below.

(1) Install SimCTG with pip.
```yaml
pip install simctg --upgrade
```

(2) Load SimCTG package with Python.
```python
import torch
# load SimCTG language model
from simctg.simctggpt import SimCTGGPT
# load SimCTG loss class
from simctg.lossfunction import SimCTGLoss
```


<span id='example_train_with_simctg'/>

##### 4.2. Example of Training Language Model with SimCTG:

<span id='init_simctg'/>

###### 4.2.1. Initialize Language Model:
```python
model_name = r'gpt2'
# initialize the language model with a vanilla GPT-2
model = SimCTGGPT(model_name)
tokenizer = model.tokenizer
```

:bell: The detailed description of SimCTGGPT can be found [[here]](https://github.com/yxuansu/SimCTG/blob/main/simctg/README.md#3-simctggpt-class).

<span id='init_loss_class'/>

###### 4.2.2. Initialize Loss Class:
```python
margin = 0.5
vocab_size = len(tokenizer)
pad_token_id = tokenizer.bos_token_id
simctgloss = SimCTGLoss(margin=margin, vocab_size=vocab_size, pad_token_id=pad_token_id)
```

:bell: The detailed description of SimCTGLoss can be found [[here]](https://github.com/yxuansu/SimCTG/blob/main/simctg/README.md#2-simctgloss-class).

**[Note]** If the margin is set as 0.0, then the SimCTG loss is equivalent to the MLE loss. 

<span id='init_training_data'/>

###### 4.2.3. Create Example Training Data:
```python
from torch.nn.utils import rnn
text_list = ['Pandas are so cute!', 'The weather in Cambridge today is very good!']
# transform batch of texts to batch of token ids
tokens_list = [tokenizer.tokenize(text) for text in text_list]
batch_id_list = [tokenizer.convert_tokens_to_ids(item) for item in tokens_list]
batch_id_list = [torch.LongTensor(item) for item in batch_id_list]
# pad the batch of token ids
batch_tensor = rnn.pad_sequence(batch_id_list, batch_first=True, padding_value=pad_token_id)
# get batch input ids and batch label ids
batch_inputs = batch_tensor[:, :-1].clone()
batch_labels = batch_tensor[:, 1:].clone()
# by setting pad token ids as -100, we stop the gradient update on these padded positions
batch_labels[batch_labels[:, :] == pad_token_id] = -100
```

<span id='compute_loss'/>

###### 4.2.4. Compute Loss:
```python
# forward computation
last_hidden_states, logits = model(input_ids=batch_inputs, labels=batch_labels)
# loss computation
mle_loss, cl_loss = simctgloss(last_hidden_states=last_hidden_states, logits=logits, 
                               input_ids=batch_inputs, labels=batch_labels)
simctg_loss = mle_loss + cl_loss
```

**[Note]** If the margin in SimCTG loss is set as 0.0, the returned cl_loss will always be 0.0 and the SimCTG loss is equivalent to the MLE loss.

<span id='contrastive_search_examples'/>

##### 4.3. Examples of Performing Generation with Contrastive Search:

<span id='example_document_generation'/>

###### 4.3.1. Open-Ended Document Generation:

We show how to reproduce our result in the case study (i.e., Table 4) of our paper.
```python
import torch
# load SimCTG language model
from simctg.simctggpt import SimCTGGPT
model_name = r'cambridgeltl/simctg_wikitext103'
model = SimCTGGPT(model_name)
model.eval()
tokenizer = model.tokenizer

# prepare input
prefix_text = r"Butt criticized Donald 's controls in certain situations in the game , as well as the difficulty of some levels and puzzles . Buchanan also criticized the controls , calling"
print ('Prefix is: {}'.format(prefix_text))
tokens = tokenizer.tokenize(prefix_text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor(input_ids).view(1,-1)

# generate result
beam_width, alpha, decoding_len = 8, 0.6, 128
output = model.fast_contrastive_search(input_ids=input_ids, beam_width=beam_width, 
                                       alpha=alpha, decoding_len=decoding_len) 
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output))
```

<details>
<summary><b>Model Output:</b></summary> 

```
Output:
----------------------------------------------------------------------------------------------------
Butt criticized Donald's controls in certain situations in the game, as well as the difficulty of some
levels and puzzles. Buchanan also criticized the controls, calling them " unimpressive " and a " nightmare "
of an experience to play with players unfamiliar with Tetris. On the other hand, his opinion was shared by
other reviewers, and some were critical of the game's technical design for the Wii version of Tetris.
In addition, Tintin's review included a quote from Roger Ebert, who said that Tetris was better than the
original game due to its simplicity and ease of play. Ebert's comments were included in the game's DVD
commentary, released on March 22, 2010. It is unclear if any of the video commentary was taken from the DVD...
```
</details>

<span id='example_dialogue_generation'/>

###### 4.3.2. Open-Domain Dialogue Generation:
We show how to reproduce our result in the case study (i.e., Table 7) of our paper.
```python
import torch
# load SimCTG language model
from simctg.simctggpt import SimCTGGPT
model_name = r'cambridgeltl/simctg_lccc_dialogue'
model = SimCTGGPT(model_name)
model.eval()
tokenizer = model.tokenizer
eos_token = '[SEP]'
eos_token_id = tokenizer.convert_tokens_to_ids([eos_token])[0]

# prepare input
context_list = ['刺猬很可爱！以前别人送了只没养，味儿太大！', '是很可爱但是非常臭', '是啊，没办法养', '那个怎么养哦不会扎手吗']
prefix_text = eos_token.join(context_list).strip(eos_token) + eos_token
print ('Prefix is: {}'.format(prefix_text))
tokens = tokenizer.tokenize(prefix_text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor(input_ids).view(1,-1)

# generate result
beam_width, alpha, decoding_len = 5, 0.6, 64
output = model.fast_contrastive_search(input_ids=input_ids, beam_width=beam_width, alpha=alpha, 
                                       decoding_len=decoding_len, end_of_sequence_token_id=eos_token_id,
                                       early_stop=True) 
print("Output:\n" + 100 * '-')
print(''.join(tokenizer.decode(output).split()))
```

<details>
<summary><b>Model Output:</b></summary> 

```
Output:
----------------------------------------------------------------------------------------------------
刺猬很可爱！以前别人送了只没养，味儿太大！[SEP]是很可爱但是非常臭[SEP]是啊，没办法养[SEP]那个怎么养哦不会扎手吗[SEP]我觉得还好，就是有点臭
```
</details>

<span id='example_off_the_shelf_generation'/>

##### 4.4. Contrastive Search with Off-the-shelf Language Models from Different Languages:
In the following, we show how to apply contrastive search on off-the-shelf language models of different languages.

<span id='chinese_example_off_the_shelf_generation'/>

###### 4.4.1. Chinese Language Model:
```python
import torch
# load SimCTG language model
from simctg.simctggpt import SimCTGGPT
model_name = r'uer/gpt2-chinese-cluecorpussmall'
model = SimCTGGPT(model_name)
model.eval()
tokenizer = model.tokenizer
eos_token = '[SEP]'
eos_token_id = tokenizer.convert_tokens_to_ids([eos_token])[0]

# Example 1
prefix_text = '苹果公司'
print ('Prefix is: {}'.format(prefix_text))
tokens = tokenizer.tokenize(prefix_text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor(input_ids).view(1,-1)

beam_width, alpha, decoding_len = 3, 0.6, 128
output = model.fast_contrastive_search(input_ids=input_ids, beam_width=beam_width, alpha=alpha, 
                                       decoding_len=decoding_len, end_of_sequence_token_id=eos_token_id,
                                       early_stop=True) 
print("Output:\n" + 100 * '-')
print(''.join(tokenizer.decode(output).split()))
```

<details>
<summary><b>Model Output:</b></summary> 

```
Output:
----------------------------------------------------------------------------------------------------
苹果公司在中国市场推出的iphone7，不仅在外观设计上有所改变，在配置上也进行了升级。苹果还宣布，新一代iphone将采用5.7英寸
屏幕，分辨率达到2560×1440像素，显示效果非常出色。此外，该机还支持指纹识别功能，可实现手指快速扫描、人脸识别等功能。
```
</details>

```python
# Example 2
prefix_text = '百节年为首，春节是中华民族最隆重的传统佳节。它不仅集中体现了中华'
print ('Prefix is: {}'.format(prefix_text))
tokens = tokenizer.tokenize(prefix_text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor(input_ids).view(1,-1)

beam_width, alpha, decoding_len = 3, 0.6, 128
output = model.fast_contrastive_search(input_ids=input_ids, beam_width=beam_width, alpha=alpha, 
                                       decoding_len=decoding_len, end_of_sequence_token_id=eos_token_id,
                                       early_stop=True) 
print("Output:\n" + 100 * '-')
print(''.join(tokenizer.decode(output).split()))
```

<details>
<summary><b>Model Output:</b></summary> 

```
Output:
----------------------------------------------------------------------------------------------------
百节年为首，春节是中华民族最隆重的传统佳节。它不仅集中体现了中华文化精髓，也表现了人民群众生活水平的提高和对美好生活的向往。
```
</details>

<span id='japanese_example_off_the_shelf_generation'/>

###### 4.4.2. Japanese Language Model:
```python
import torch
# load SimCTG language model
from simctg.simctggpt import SimCTGGPT
model_name = r'colorfulscoop/gpt2-small-ja'
model = SimCTGGPT(model_name)
model.eval()
tokenizer = model.tokenizer
eos_token = tokenizer.eos_token
eos_token_id = tokenizer.convert_tokens_to_ids([eos_token])[0]

# prepare input
prefix_text = r'臥龍桜（がりゅうざくら）は、岐阜県高山市一之宮町にある一本桜。龍が地'
print ('Prefix is: {}'.format(prefix_text))
tokens = tokenizer.tokenize(prefix_text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor(input_ids).view(1,-1)

# generate result
beam_width, alpha, decoding_len = 5, 0.6, 128
output = model.fast_contrastive_search(input_ids=input_ids, beam_width=beam_width, alpha=alpha, 
                                       decoding_len=decoding_len, end_of_sequence_token_id=eos_token_id,
                                       early_stop=True) 
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output))
```

<details>
<summary><b>Model Output:</b></summary> 

```
Output:
----------------------------------------------------------------------------------------------------
臥龍桜(がりゅうざくら)は、岐阜県高山市一之宮町にある一本桜。龍が地中に染みつく様子を図案化したもので、樹齢400年を越す日本
さくら名所100選に選定されている。一之宮町指定天然記念物。岐阜県飛騨地方(東濃地方)の山間地に生育し、約1万年前に絶滅したと
考えられている。「花の本」とも称され、開花期は5月上旬から下旬までで、桜の枝張りは濃緑色である。花は直径約10cmの花弁を咲か
せる八重咲きで、花弁の色は紅紫色で、雄しべは4本、雌しべは1本ある。雄しべの先
```
</details>

<span id='korean_example_off_the_shelf_generation'/>

###### 4.4.3. Korean Language Model:
```python
import torch
# load SimCTG language model
from simctg.simctggpt import SimCTGGPT
model_name = r'skt/ko-gpt-trinity-1.2B-v0.5'
model = SimCTGGPT(model_name)
model.eval()
tokenizer = model.tokenizer
eos_token = tokenizer.eos_token
eos_token_id = tokenizer.convert_tokens_to_ids([eos_token])[0]

# prepare input
prefix_text = r'인간처럼 생각하고, 행동하는 \'지능\'을 통해 인류가 이제까지 풀지 못했던'
print ('Prefix is: {}'.format(prefix_text))
tokens = tokenizer.tokenize(prefix_text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor(input_ids).view(1,-1)

# generate result
beam_width, alpha, decoding_len = 5, 0.6, 64
output = model.fast_contrastive_search(input_ids=input_ids, beam_width=beam_width, alpha=alpha, 
                                       decoding_len=decoding_len, end_of_sequence_token_id=eos_token_id,
                                       early_stop=True) 
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output))
```

<details>
<summary><b>Model Output:</b></summary> 

```
Output:
----------------------------------------------------------------------------------------------------
인간처럼 생각하고, 행동하는 \'지능\'을 통해 인류가 이제까지 풀지 못했던 난제를 해결하려 한다. 이 책의 제목이기도 한 '슈퍼인텔리전스'는 
인공지능(AI)의 등장으로 야기된 사회 변화를 일컫는 말로, 이 책을 관통하는 키워드이기도 하다. 저자는 "기술과 인간 사이의 경계가 무너지고 
있다"고 지적한다. AI가 인간의 사고방식과 행동을 모방할 뿐만
```
</details>


<span id='training_tutorial'/>

##### 4.5. Detailed Tutorial of Training SimCTG on Wikitext-103: 
We also provide a comprehensive tutorial on how to reproduce our experiments on Wikitext-103 using the released package. Check it [[here]](https://github.com/yxuansu/SimCTG/tree/main/training_tutorial_on_wikitext103)!


<span id='T5_tutorial'/>

##### 4.6. Apply SimCTG on T5:
We also provide detailed tutorials on how to apply SimCTG and contrastive search on T5 model. For more details, please refer to [[here]](https://github.com/yxuansu/SimCTG/tree/main/simctg#4-simctgt5-class-back-to-top) and [[here]](https://github.com/yxuansu/SimCTG/blob/main/SimCTGEncDec/README.md#2-t5-back-to-top).


****

<span id='environment_setup'/>

#### 5. Environment Setup: <a href='#all_catelogue'>[Back to Top]</a>
```yaml
python version >= 3.6
pip3 install -r requirements.txt
```

****

:exclamation::exclamation::exclamation: **[Note]** The following instructions were originally used in the experiments of our paper. Now we have provided an easy-to-use library which helps you to implement SimCTG with just a few lines of code (**Of course, the original code still works!**). Check it [[here]](#tutorial)!

****
<span id='example_usage'/>

#### 6. Example Usage of Contrastive Search: <a href='#all_catelogue'>[Back to Top]</a>

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
```
<details>
<summary><b>Model Output:</b></summary>
  
```yaml
Insect farming is the practice of raising and breeding insects as livestock, also referred to as minilivestock
or micro stock. Insects may be farmed for the  commodities they produce, such as honey, corn, sorghum, and 
other crops. In some cases, the production of insects is a way to increase income for the owner or his family. 
This type of farming has been described as "an economic system that benefits all people regardless of race, sex, 
or social status" (p.\xa09). A large number of farmers in North America, Europe, and South America have used the 
method of farming for food production in order to feed their families and livestock. The most common method of 
farming is by hand-cropping, which consists of cutting a hole in the ground and using a saw
```
</details>


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
```

<details>
<summary><b>Model Output:</b></summary> 

```
苹果公司在中国市场推出的iphone7，不仅在外观设计上有所改变，在配置上也进行了升级。苹果还宣布，新一代iphone将采用
5.7英寸屏幕，分辨率达到2560×1440像素，显示效果非常出色。此外，该机还支持指纹识别功能，可实现手指快速扫描、人脸识
别等功能。
```
</details>

```python 
# (2) use nucleus sampling to generate the result
nucleus_p, decoding_len = 0.95, 128
eos_token = '[SEP]'
print (model.nucleus_sampling(input_ids, nucleus_p, decoding_len, eos_token))
```

<details>
<summary><b>Model Output:</b></summary> 

```
苹果公司的设计套件。2.不同的颜色设计有不同的热塑性材质。热塑性材质中的ca34bc是真正能够让人感觉舒适的材质。3.比利
时家具建筑师埃莉诺特·夏格和大家举一些非常实用又非常普遍的例子在这里艾格的设计师们会简单介绍一下为什么美国家具是比利
时建筑的一个分支或一个分支，他们
```
</details>

```python 
# (3) use greedy search to generate the result
decoding_len = 128
eos_token = '[SEP]'
print (model.greedy_search(input_ids, decoding_len, eos_token))
```

<details>
<summary><b>Model Output:</b></summary> 

```
苹果公司的一个重要客户，他们的产品在全球范围内都有着非常高的知名度。[UNK]我们的产品在全球范围内都有着非常高的知名度，
我们的产品在全球范围内都有着非常高的知名度。[UNK]在这样的背景下，苹果公司的产品在全球范围内都有着非常高的知名度。[UNK]
我们的产品在全球范围内都有着非常高的知
```
</details>

```python 
# (4) use beam search to generate the result
beam_width, decoding_len = 10, 128
eos_token = '[SEP]'
print (model.beam_search(input_ids, 10, decoding_len, eos_token))
```

<details>
<summary><b>Model Output:</b></summary> 

```
苹果公司总裁兼首席执行官蒂姆·库克（timcook）表示：[UNK]苹果公司是全球最大的智能手机制造商之一，苹果公司是全球最大的
智能手机制造商之一，苹果公司是全球最大的智能手机制造商之一，苹果公司是全球最大的智能手机制造商之一，苹果公司是全球最大
的智能手机制造商之一，苹果公司是全球
```
</details>


```python
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
```

<details>
<summary><b>Model Output:</b></summary> 

```
百节年为首，春节是中华民族最隆重的传统佳节。它不仅集中体现了中华文化精髓，也表现了人民群众生活水平的提高和对美好生活的向往。
```
</details>

```python
# (2) use nucleus sampling to generate the result
nucleus_p, decoding_len = 0.95, 128
eos_token = '[SEP]'
print (model.nucleus_sampling(input_ids, nucleus_p, decoding_len, eos_token))
```

<details>
<summary><b>Model Output:</b></summary> 

```
百节年为首，春节是中华民族最隆重的传统佳节。它不仅集中体现了中华传统文化，更是经济、政治、文化上的一个精神机能的全面发展。
人们在生活中不仅能够充分认识到这个民族的非物质文化遗产，而且能够在此基础上追求书面化的概念。中国历史上有许多著名的「人物」
，他们深深地扎根于中国历史的传统历史文化中，热爱中华文化，热爱中华文化的传承
```
</details>

```python
# (3) use greedy search to generate the result
decoding_len = 128
eos_token = '[SEP]'
print (model.greedy_search(input_ids, decoding_len, eos_token))
```

<details>
<summary><b>Model Output:</b></summary> 

```
百节年为首，春节是中华民族最隆重的传统佳节。它不仅集中体现了中华民族的传统美德，也体现了中华民族的传统文化。[UNK]中华民族
的传统美德，是中华民族的传统美德。[UNK]中华民族的传统美德，是中华民族的传统美德。[UNK]中华民族的传统美德，是中华民族的传
统美德。[UNK]中华民族的传统美德，是中华民族的传统美德。[UNK]中华民族的传统美德，是中华民族的传
```
</details>

```python
# (4) use beam search to generate the result
beam_width, decoding_len = 10, 128
eos_token = '[SEP]'
print (model.beam_search(input_ids, 10, decoding_len, eos_token))
```

<details>
<summary><b>Model Output:</b></summary> 

```
百节年为首，春节是中华民族最隆重的传统佳节。它不仅集中体现了中华民族伟大复兴的历史使命，也体现了中华民族伟大复兴的历史使命。
中华民族伟大复兴的历史使命，不仅体现了中华民族伟大复兴的历史使命，也体现了中华民族伟大复兴的历史使命。中华民族伟大复兴的历
史使命，不仅体现了中华民族伟大复兴的历史使命，也体现了中华民族伟大复兴的历
```
</details>

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
```

<details>
<summary><b>Model Output:</b></summary> 

```
臥龍桜(がりゅうざくら)は、岐阜県高山市一之宮町にある一本桜。龍が地中に染みつく様子を図案化したもので、樹齢400年
を越す日本さくら名所100選に選定されている。一之宮町指定天然記念物。岐阜県飛騨地方(東濃地方)の山間地に生育し、約
1万年前に絶滅したと考えられている。「花の本」とも称され、開花期は5月上旬から下旬までで、桜の枝張りは濃緑色である。
花は直径約10cmの花弁を咲かせる八重咲きで、花弁の色は紅紫色で、雄しべは4本、雌しべは1本ある。雄しべの先
```
</details>



```python
# (2) use nucleus sampling to generate the result
nucleus_p, decoding_len = 0.95, 128
eos_token = model.tokenizer.eos_token
print (model.nucleus_sampling(input_ids, nucleus_p, decoding_len, eos_token))
```


<details>
<summary><b>Model Output:</b></summary> 

```
臥龍桜(がりゅうざくら)は、岐阜県高山市一之宮町にある一本桜。龍が地中に棲む奇岩に由来する。毎年5月上旬には多くの花見
客が訪れている。かつて、雪見の藩お抱え家臣、雲口である長久城主長久竜泰が祭っている「月輪寺」には手水鉢が2つあり、長
久氏の勢力が強まると同時に関連する寺もあり、山を挟むように吉野側の赤峰山から北へ順に樹齢250年を越してきたが、江戸時
代に廃材が搬出されてから薪が取れなくなっている。古い株は毎年12月の初午に燃えつき風雨が吹き荒れて朽ち果てる。根は分枝
```
</details>


```python
# (3) use greedy search to generate the result
decoding_len = 128
eos_token = model.tokenizer.eos_token
print (model.greedy_search(input_ids, decoding_len, eos_token))
```

<details>
<summary><b>Model Output:</b></summary> 

```
臥龍桜(がりゅうざくら)は、岐阜県高山市一之宮町にある一本桜。龍が地中に棲む龍の棲むとされる桜で、樹齢は1000年以上。樹
高は10mほどで、幹周りは8mほどになる。樹齢は300年ほどで、樹高は20mほどになる。樹形が整っており、枝張りも良く、樹勢も
旺盛である。樹形は、樹高が1mほどで、幹周りは4mほどになる。枝張りはよく発達し、樹勢は旺盛である。冬になると、幹周りの
樹冠が紅葉する。また、紅葉の時期には、樹冠が赤く紅葉する。樹
```
</details>


```python
# (4) use beam search to generate the result
beam_width, decoding_len = 10, 128
eos_token = model.tokenizer.eos_token
print (model.beam_search(input_ids, 10, decoding_len, eos_token))
```

<details>
<summary><b>Model Output:</b></summary> 

```
臥龍桜(がりゅうざくら)は、岐阜県高山市一之宮町にある一本桜。龍が地中深くに咲く桜で、岐阜県の天然記念物に指定されている。
岐阜県高山市一之宮町一之宮(いちのみやちょういちのみや)は、岐阜県高山市一之宮町一之宮にある一本桜である。龍が地中深くに
咲く桜で、岐阜県の天然記念物に指定されている。岐阜県高山市一之宮町一之宮(いちのみやちょういちのみや)は、岐阜県高山市一
之宮町一之宮(いちのみやちょういちのみや)は、岐阜県高山市一之宮町一之宮(いちのみやちょういちのみや)は、岐阜県高山
```
</details>

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
```

<details>
<summary><b>Model Output:</b></summary> 

```
인간처럼생각하고,행동하는\'지능\'을통해인류가이제까지풀지못했던난제를해결하려한다.이책의제목이기도한'슈퍼인텔리전스'는인공지능
(AI)의등장으로야기된사회변화를일컫는말로,이책을관통하는키워드이기도하다.저자는"기술과인간사이의경계가무너지고있다"고지적한다.
AI가인간의사고방식과행동을모방할뿐만
```
</details>

```python
# (2) use nucleus sampling to generate the result
nucleus_p, decoding_len = 0.95, 64
eos_token = model.tokenizer.eos_token
print (model.nucleus_sampling(input_ids, nucleus_p, decoding_len, eos_token))
```

<details>
<summary><b>Model Output:</b></summary> 

```
인간처럼생각하고,행동하는\'지능\'을통해인류가이제까지풀지못했던큰수수께끼를풀수있다.'지능\'은인공두뇌그자체이기도하지만그공간의
반영이라는해석도가능하다.예를들면시간부등호처럼복잡한수식을쉽게떠올릴수있다는이야기다.마치구글에검색창에'Quick'이라는단어를입력하
면자동으로'중력'은일정한법칙에따라
```
</details>

```python
# (3) use greedy search to generate the result
decoding_len = 64
eos_token = model.tokenizer.eos_token
print (model.greedy_search(input_ids, decoding_len, eos_token))
```

<details>
<summary><b>Model Output:</b></summary> 

```
인간처럼생각하고,행동하는\'지능\'을통해인류가이제까지풀지못했던문제를해결할수있다고주장한다.이지능은\'지능\'그자체라기보다\'지능\'
그자체를구성하는\'지능\'그자체라고할수있다.이지능은\'지능\'그자체라기보다\'지능\'그자체를구성하는\'지능\'그자체라고
```
</details>


```python
# (4) use beam search to generate the result
# We do not print the result, because beam search stops generation immediately.
```

**[Note]** Sadly, I am not a Korean speaker either, so I can only judge the quality of the generated text using [Google translate](https://translate.google.com/) as well. It would be great if anyone could tell me whether the generated text is good or not. Thank you!

****

<span id='wikitext103_tutorial'/>

:exclamation::exclamation::exclamation: **[Note]** The following instructions were originally used in the experiments of our paper. Now we have provided an easy-to-use library which helps you to implement SimCTG with just a few lines of code (**Of course, the original code still works!**). Check it <a href='#tutorial'>[here]</a>!

#### 7. Document Generation: <a href='#all_catelogue'>[Back to Top]</a>
The detailed tutorial of experiment on document generation is provided [[here]](https://github.com/yxuansu/SimCTG/tree/main/document_generation).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_zPZRlbJo5iw_Q7FUhP113udPnciUxVF?usp=sharing)

****

<span id='dialogue_tutorial'/>

:exclamation::exclamation::exclamation: **[Note]** The following instructions were originally used in the experiments of our paper. Now we have provided an easy-to-use library which helps you to implement SimCTG with just a few lines of code (**Of course, the original code still works!**). Check it <a href='#tutorial'>[here]</a>!  

#### 8. Open-domain Dialogue Generation: <a href='#all_catelogue'>[Back to Top]</a>
The detailed tutorial of experiment on open-domain dialogue generation provided [[here]](https://github.com/yxuansu/SimCTG/tree/main/dialogue_generation).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/drive/1_55LEg2caLM-lYDVIhWjxgv75IWkEry6?usp=sharing)

****

<span id='pretraining'/>

#### 9. Large-Scale Pre-training with SimCTG: <a href='#all_catelogue'>[Back to Top]</a>
In addition to fine-tuning on downstream tasks (e.g. document generation and open-domain dialogue generation), we can also use a large-scale general domain corpus (i.e. Wikipedia) to pre-train a SimCTG model. [Here](https://github.com/yxuansu/SimCTG/tree/main/pretraining), we show the details of how to pre-train SimCTG using a large-scale English Wikipedia corpus.


****

<span id='story_generation'/>

#### 10. Open-Ended Story Generation: <a href='#all_catelogue'>[Back to Top]</a>
We also show how to adapt our approach to open-ended story generation task. The details are provided [[here]](https://github.com/yxuansu/SimCTG/tree/main/story_generation).


****

<span id='contrastive_for_encoder_decoder'/>

#### 11. Contrastive Search on Encoder-Decoder Models: <a href='#all_catelogue'>[Back to Top]</a>

Details on how to apply contrastive search on encoder-decoder models (e.g. BART and T5) can be found [[here]](https://github.com/yxuansu/SimCTG/tree/main/SimCTGEncDec).

****

<span id='contact'/>

#### 12. Contact: <a href='#all_catelogue'>[Back to Top]</a>
If you have any questions, feel free to contact me via (ys484 at cam.ac.uk).

****

<span id='simctg_elsewhere'/>

#### 13. SimCTG Elsewhere: <a href='#all_catelogue'>[Back to Top]</a>

We thank the community's effort for extending SimCTG!

- [Zenn](https://zenn.dev/) has provided [a tutorial and implementation of contrastive search based on T5](https://zenn.dev/kwashizzz/articles/ml-simctg-contrastive-framework). 
- [Tencent's Effidit platform (腾讯智能创作助手)](https://effidit.qq.com/) has integrated SimCTG into its AI sentence completion module. Check it out and have fun! 

