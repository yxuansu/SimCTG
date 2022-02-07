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
To download the data, please follow the instructions [here](https://github.com/yxuansu/SimCTG/tree/main/data).

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
To train a SimCTG model on Wikitext-103, please run the following commands:
```yaml
chmod +x ./train.sh
./train.sh
```
The arguments are as follows:
* `--model_name`: The name of huggingface pre-trained gpt model (e.g. gpt2, gpt-large).
* `--train_path`: The file path of training set.
* `--dev_path`: The file path of validation set.
* `--test_path`: The file path of test set.
* `--margin`: The contrastive margin $\rho$.
* `--min_len`: The minimum length of training samples.
* `--max_len`: The maximum length of training samples.
* `--eos_token`: The special token used to indicate the boundary between different utterances.
* `--pad_token`: The special token used to pad the training batch during the training process.
* `--max_len`: The maximum length of training samples.
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
Here, we show how to use SimCTG to generate dialogue response with different decoding methods.
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
```
<span id='contrastive_search'/>

##### 3.1. Contrastive Search:
```python
# use contrastive search to generate the result
beam_width, alpha, decoding_len = 8, 0.6, 128
output = model.fast_contrastive_search(input_ids, beam_width, alpha, decoding_len)
#output = model.slow_contrastive_search(input_ids, beam_width, alpha, decoding_len)
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output))
```
The arguments are as follows:
* `--input_ids`: The ids of the prefix sequence.
* `--beam_width`: k in the contrastive search, which is typically set within the range of [5,10].
* `--alpha`: alpha in the contrastive search, which is typically set within the range of [0.5,0.8].
* `--decoding_len`: Number of tokens to generate.

**[Note]** We provide two implementations of contrastive search: (1) fast_contrastive_search and (2) slow_contrastive_search. These two implementations produce the same result, but the fast version is properly optimized and is much faster than the slow version. On the other hand, the implementation details of the slow version is more straightforward. We recommend you to rewrite the slow version first if you want to adapt contrastive search to your specific research task.

<span id='diverse_contrastive_search'/>

##### 3.2. Diverse Contrastive Search:
We also provide a stochastic version of contrastive search which can generate diverse results by combining nucleus sampling and contrastive search. More details can be found in Appendix E of the [paper]().
```python
# use diverse contrastive search to generate the result
sample_step, nucleus_p = 2, 0.95
beam_width, alpha, decoding_len = 8, 0.6, 128
output = model.diverse_contrastive_search(input_ids, sample_step, nucleus_p, beam_width, alpha, decoding_len)
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output))
```
The arguments are as follows:
* `--input_ids`: The ids of the prefix sequence.
* `--sample_step`: The number of tokens sampled with nucleus sampling at the start of generation process.
* `--nucleus_p`: The probability in nucleus sampling.
* `--beam_width`: k in the contrastive search, which is typically set within the range of [5,10].
* `--alpha`: alpha in the contrastive search, which is typically set within the range of [0.5,0.8].
* `--decoding_len`: The total number of tokens to generate. It equals to the number of sampled tokens + the number of tokens generated by contrastive search.

<span id='nucleus_sampling'/>

##### 3.3. Nucleus Sampling:
```python
nucleus_p, decoding_len = 0.95, 128
output = model.nucleus_sampling(input_ids, nucleus_p, decoding_len)
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output))
```
The arguments are as follows:
* `--input_ids`: The ids of the prefix sequence.
* `--nucleus_p`: The probability in nucleus sampling.
* `--decoding_len`: Number of tokens to generate.

<span id='greedy_search'/>

##### 3.4. Greedy Search:
```python
decoding_len = 128
output = model.greedy_search(input_ids, decoding_len)
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output))
```
The arguments are as follows:
* `--input_ids`: The ids of the prefix sequence.
* `--decoding_len`: Number of tokens to generate.

<span id='beam_search'/>

##### 3.5. Beam Search:
```python
beam_width, decoding_len = 10, 128
output = model.beam_search(input_ids, beam_width, decoding_len)
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output))
```
The arguments are as follows:
* `--input_ids`: The ids of the prefix sequence.
* `--beam_width`: The beam width of beam search.
* `--decoding_len`: Number of tokens to generate.


