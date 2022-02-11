## This repo describes how to pre-train SimCTG on a large-scale pre-training corpus.
****
### Catalogue:
* <a href='#data_preparation'>1. Data Preparation</a>
* <a href='#train_simctg'>2. Train SimCTG</a>
* <a href='#generate_results'>3. Generate Result with English SimCTG using Different Decoding Methods</a>
    * <a href='#contrastive_search'>3.1. Contrastive Search</a>
    * <a href='#diverse_contrastive_search'>3.2. Diverse Contrastive Search</a>
    * <a href='#nucleus_sampling'>3.3. Nucleus Sampling</a>
    * <a href='#greedy_search'>3.4. Greedy Search</a>
    * <a href='#beam_search'>3.5. Beam Search</a>


****
<span id='data_preparation'/>

#### 1. Data Preparation:
To download the pre-training the large-scale Wikipedia corpus, please follow the instructions [[here]](https://github.com/yxuansu/SimCTG/tree/main/data).

> **** The dataset contains the following files:

    .
    ├── wikipedia                   
        ├── train_english_wikipedia.txt       # Training set
        └── dev_english_wikipedia.txt         # Validation set

**Data Format**: In the files, each line represents a complete wikipedia article.

****

<span id='train_simctg'/>

#### 2. Train SimCTG:
To pre-train SimCTG on Wikipedia corpus, please run the following commands:
```yaml
chmod +x ./train.sh
./train.sh
```
The arguments are as follows:
* `--model_name`: The name of huggingface pre-trained gpt model (e.g. gpt2).
* `--train_path`: The file path of training set.
* `--dev_path`: The file path of validation set.
* `--margin`: The contrastive margin $\rho$.
* `--seqlen`: The length of every training sample.
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

#### 3. Generate Result with English SimCTG using Different Decoding Methods:
Here, we show how to use SimCTG to generate open-ended document with different decoding methods.
```python
import torch
from simctg import SimCTGPretraining
model_path = r'cambridgeltl/simctg_english_wikipedia'
model = SimCTGPretraining(model_path)
model.eval()

# prepare text prefix
text = r'Insect farming is the practice of raising and breeding insects as livestock, also referred to as minilivestock or micro stock. Insects may be farmed for the commodities'
tokens = model.tokenizer.tokenize(text)
input_ids = model.tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor(input_ids).view(1,-1)

```
<span id='contrastive_search'/>

##### 3.1. Contrastive Search:
```python
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
The arguments are as follows:
* `--input_ids`: The ids of the prefix sequence.
* `--beam_width`: k in the contrastive search, which is typically set within the range of [3,10].
* `--alpha`: alpha in the contrastive search, which is typically set within the range of [0.5,0.8].
* `--decoding_len`: Number of tokens to generate.
* `--eos_token`: The token that indicates the end of sequence.

<span id='diverse_contrastive_search'/>

##### 3.2. Diverse Contrastive Search:
We also provide a stochastic version of contrastive search which can generate diverse results by combining nucleus sampling and contrastive search. More details can be found in the Appendix G of our [paper]().
```python
# use diverse contrastive search to generate the result
beam_width, alpha, decoding_len = 5, 0.6, 128
eos_token = '<|endoftext|>'
sample_step, nucleus_p = 2, 0.95
print (model.diverse_contrastive_search(input_ids, sample_step, nucleus_p, beam_width, alpha, decoding_len, eos_token))
```
The arguments are as follows:
* `--input_ids`: The ids of the prefix sequence.
* `--sample_step`: The number of tokens sampled with nucleus sampling at the start of generation process.
* `--nucleus_p`: The probability in nucleus sampling.
* `--beam_width`: k in the contrastive search, which is typically set within the range of [3,10].
* `--alpha`: alpha in the contrastive search, which is typically set within the range of [0.5,0.8].
* `--decoding_len`: The total number of tokens to generate. It equals to the number of sampled tokens + the number of tokens generated by contrastive search.
* `--eos_token`: The token that indicates the end of sequence.

<span id='nucleus_sampling'/>

##### 3.3. Nucleus Sampling:
```python
nucleus_p, decoding_len = 0.95, 128
print (model.nucleus_sampling(input_ids, nucleus_p, decoding_len, eos_token))
```
The arguments are as follows:
* `--input_ids`: The ids of the prefix sequence.
* `--nucleus_p`: The probability in nucleus sampling.
* `--decoding_len`: Number of tokens to generate.
* `--eos_token`: The token that indicates the end of sequence.

<span id='greedy_search'/>

##### 3.4. Greedy Search:
```python
decoding_len = 128
print (model.greedy_search(input_ids, decoding_len, eos_token))
```
The arguments are as follows:
* `--input_ids`: The ids of the prefix sequence.
* `--decoding_len`: Number of tokens to generate.
* `--eos_token`: The token that indicates the end of sequence.

<span id='beam_search'/>

##### 3.5. Beam Search:
```python
beam_width, decoding_len = 10, 128
print (model.beam_search(input_ids, beam_width, decoding_len, eos_token))
```
The arguments are as follows:
* `--input_ids`: The ids of the prefix sequence.
* `--beam_width`: The beam width of beam search.
* `--decoding_len`: Number of tokens to generate.
* `--eos_token`: The token that indicates the end of sequence.


