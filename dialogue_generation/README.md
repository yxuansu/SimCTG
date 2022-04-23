## This repo describes the experimental details on LCCC benchmark.
****
### Catalogue:
* <a href='#data_preparation'>1. Data Preparation</a>
* <a href='#train_simctg'>2. Train SimCTG</a>
* <a href='#generate_results'>3. Generate Result with Different Decoding Methods</a>
    * <a href='#contrastive_search'>3.1. Contrastive Search</a>
    * <a href='#diverse_contrastive_search'>3.2. Diverse Contrastive Search</a>
    * <a href='#nucleus_sampling'>3.3. Nucleus Sampling</a>
    * <a href='#greedy_search'>3.4. Greedy Search</a>
    * <a href='#beam_search'>3.5. Beam Search</a>


****
<span id='data_preparation'/>

#### 1. Data Preparation:
To download the data, please follow the instructions [[here]](https://github.com/yxuansu/SimCTG/tree/main/data).

> **** The dataset contains the following files:

    .
    ├── LCCC                     
        ├── LCCC-base_train.txt        # Training set
        ├── LCCC-base_train_1_million_lines.txt     # File contains the first one million samples of the full training set
        ├── LCCC-base_dev.txt          # Validation set
        └── LCCC-base_test.txt         # Test set

**Data Format**: In the files, each line represents a complete dialogue session, where the utterances are seperated by '\t'.

****

<span id='train_simctg'/>

#### 2. Train SimCTG:
To train a SimCTG model on LCCC dataset, please run the following commands:
```yaml
chmod +x ./train.sh
./train.sh
```
The arguments are as follows:
* `--model_name`: The name of huggingface pre-trained gpt model (e.g. gpt2 and uer/gpt2-chinese-cluecorpussmall).
* `--train_path`: The file path of training set.
* `--dev_path`: The file path of validation set.
* `--test_path`: The file path of test set.
* `--margin`: The contrastive margin $\rho$.
* `--min_len`: The minimum length of training samples.
* `--max_len`: The maximum length of training samples.
* `--eos_token`: The special token used to indicate the boundary between different utterances. For Chinese and English dataset, use '[SEP]' and 'endoftext', respectively.
* `--pad_token`: The special token used to pad the training batch during the training process.
* `--number_of_gpu`: The number of available GPUs.
* `--batch_size_per_gpu`: The batch size for each GPU.
* `--gradient_accumulation_steps`: How many forward computations between two gradient updates.
* `--effective_batch_size`: The overall batch size. It equals to batch_size_per_gpu x gradient_accumulation_steps x number_of_gpu.
* `--total_steps`: The number of total gradient update steps.
* `--print_every`: Have many steps to show the intermediate results.
* `--save_every`: How many steps to save one checkpoint.
* `--learning_rate`: The learning rate.
* `--save_path_prefix`: Where to save the checkpoints.

****
<span id='generate_results'/>

#### 3. Generate Result with Different Decoding Methods:
Here, we show how to use SimCTG to generate dialogue response with different decoding methods. More generated examples can be found in the Appendix D of our [paper](https://arxiv.org/abs/2202.06417).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/drive/1_55LEg2caLM-lYDVIhWjxgv75IWkEry6?usp=sharing)

```python
import torch
from simctgdialogue import SimCTGDialogue
# load model
model_path = r'cambridgeltl/simctg_lccc_dialogue'
eos_token, pad_token = '[SEP]', '[PAD]'
model = SimCTGDialogue(model_path, eos_token, pad_token)
tokenizer = model.tokenizer
model.eval()

# prepare the dailogue context which is a list of utterances
context_list = ['刺猬很可爱！以前别人送了只没养，味儿太大！', '是很可爱但是非常臭', '是啊，没办法养', '那个怎么养哦不会扎手吗']
```
<span id='contrastive_search'/>

##### 3.1. Contrastive Search:
```python
# use contrastive search to generate the result
beam_width, alpha, decoding_len = 5, 0.6, 64
print (model.contrastive_search(context_list, beam_width, alpha, decoding_len))
# '我觉得还好，就是有点臭'
```
The arguments are as follows:
* `--context_list`: A list of utterances, e.g. [utterance_1, utterance_2, ..., utterance_n].
* `--beam_width`: k in the contrastive search, which is typically set within the range of [3,10].
* `--alpha`: alpha in the contrastive search, which is typically set within the range of [0.5,0.8].
* `--decoding_len`: Number of tokens to generate.

<span id='diverse_contrastive_search'/>

##### 3.2. Diverse Contrastive Search:
We also provide a stochastic version of contrastive search which can generate diverse results by combining nucleus sampling and contrastive search. More details can be found in Appendix G of the [paper]().
```python
# use diverse contrastive search to generate the result
sample_step, nucleus_p = 1, 0.95
beam_width, alpha, decoding_len = 5, 0.6, 64
print (model.diverse_contrastive_search(context_list, sample_step, nucleus_p, beam_width, alpha, decoding_len))
# '可以的，就是有点疼'
```
The arguments are as follows:
* `--context_list`: A list of utterances, e.g. [utterance_1, utterance_2, ..., utterance_n].
* `--sample_step`: The number of tokens sampled with nucleus sampling at the start of generation process.
* `--nucleus_p`: The probability in nucleus sampling.
* `--beam_width`: k in the contrastive search, which is typically set within the range of [3,10].
* `--alpha`: alpha in the contrastive search, which is typically set within the range of [0.5,0.8].
* `--decoding_len`: The total number of tokens to generate. It equals to the number of sampled tokens + the number of tokens generated by contrastive search.

<span id='nucleus_sampling'/>

##### 3.3. Nucleus Sampling:
```python
nucleus_p, decoding_len = 0.95, 64
print (model.nucleus_sampling(context_list, nucleus_p, decoding_len))
# '我是在家里养的感觉不会弄它血的特别快啊我之前也怕它血不会流很多，但是你家都这么养的可能都被它养着我家在南京'
```
The arguments are as follows:
* `--context_list`: A list of utterances, e.g. [utterance_1, utterance_2, ..., utterance_n].
* `--nucleus_p`: The probability in nucleus sampling.
* `--decoding_len`: Number of tokens to generate.

<span id='greedy_search'/>

##### 3.4. Greedy Search:
```python
decoding_len = 64
print (model.greedy_search(context_list, decoding_len))
# '我也不知道，我家的也是，我家的也是，我家的也是，我家的也是，我家的也是，我家的也是，我家的也是，我家的也是，我家的也是，我家的也'
```
The arguments are as follows:
* `--context_list`: A list of utterances, e.g. [utterance_1, utterance_2, ..., utterance_n].
* `--decoding_len`: Number of tokens to generate.

<span id='beam_search'/>

##### 3.5. Beam Search:
```python
beam_width, decoding_len = 10, 64
print (model.beam_search(context_list, beam_width, decoding_len))
# '可以的'
```
The arguments are as follows:
* `--context_list`: A list of utterances, e.g. [utterance_1, utterance_2, ..., utterance_n].
* `--beam_width`: The beam width of beam search.
* `--decoding_len`: Number of tokens to generate.


