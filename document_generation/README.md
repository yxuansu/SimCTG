## This repo describes the experimental details on Wikitext-103 benchmark.
****
### Catalogue:
* <a href='#data_preparation'>1. Data Preparation</a>
* <a href='#train_simctg'>2. Train SimCTG</a>
* <a href='#inference'>3. Inference with SimCTG</a>
* <a href='#generate_results'>4. Generate Result with Different Decoding Methods</a>
    * <a href='#contrastive_search'>4.1. Contrastive Search</a>
    * <a href='#diverse_contrastive_search'>4.2. Diverse Contrastive Search</a>
    * <a href='#nucleus_sampling'>4.3. Nucleus Sampling</a>
    * <a href='#greedy_search'>4.4. Greedy Search</a>
    * <a href='#beam_search'>4.5. Beam Search</a>
* <a href='#evaluation'>5. Evaluate the Generated Text</a>
* <a href='#visualize_token_similarity_matrix'>6. Visualize the Token Similarity Matrix</a>


****
<span id='data_preparation'/>

#### 1. Data Preparation:
To download the data, please follow the instructions [[here]](https://github.com/yxuansu/SimCTG/tree/main/data).

> **** The dataset contains the following three files:

    .
    ├── wikitext103                       
        ├── wikitext103_raw_v1_train.txt          # Training Set
        ├── wikitext103_raw_v1_validation.txt     # Validation Set
        └── wikitext103_raw_v1_test.txt           # Test Set

**Data Format**: In the files, each line represents a full document.

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

<span id='inference'/>

#### 3. Inference with SimCTG
Here we show how to use SimCTG to perform inference with prefixes from validation and test sets.
```yaml
chmod +x ./inference.sh
./inference.sh
```
The arguments are as follows:
* `--ckpt_path`: The path of trained checkpoint. You can either use our released checkpoint (`cambridgeltl/simctg_wikitext103`) or your own trained model that can be found in the `--save_path_prefix` directory as defined in train.sh.
* `--dev_path`: The file path of validation set.
* `--test_path`: The file path of test set.
* `--prefix_len`: The length of prefix.
* `--decoding_len`: The length of generated text continuation.
* `--k`: The k in contrastive search.
* `--alpha`: The \alpha in contrastive search.
* `--save_path`: Where to save the generated result.

The generated file is a list of dictionary, where the data format of each dictionary is:

```yaml
{  
   "prefix_text": The human-written prefix.
   "reference_text": The reference document (prefix + reference text continuation).
   "reference_continuation_text": The reference text continuation.   
   "generated_result": {
       "0": {
           "full_text": The prefix + generated continuation.
           "continuation": The generated continuation.
            }
       }
}
```

**[Note]** We provide our generated file in ./simctg_contrasive.json.

****
<span id='generate_results'/>

#### 4. Generate Result with Different Decoding Methods:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_zPZRlbJo5iw_Q7FUhP113udPnciUxVF?usp=sharing)

Here, we use the prefix in Table 4 of our [paper](https://arxiv.org/abs/2202.06417) to illustrate how to use different decoding methods to generate the result. 
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

##### 4.1. Contrastive Search:
```python
# use contrastive search to generate the result
beam_width, alpha, decoding_len = 8, 0.6, 128
output = model.fast_contrastive_search(input_ids, beam_width, alpha, decoding_len)
#output = model.slow_contrastive_search(input_ids, beam_width, alpha, decoding_len)
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output))

'''
   Butt criticized Donald's controls in certain situations in the game, as well as 
   the difficulty of some levels and puzzles. Buchanan also criticized the controls, 
   calling them " unimpressive " and a " nightmare " of an experience to play with 
   players unfamiliar with Tetris. On the other hand, his opinion was shared by other 
   reviewers, and some were critical of the game's technical design for the Wii version 
   of Tetris. In addition, Tintin's review included a quote from Roger Ebert, who said 
   that Tetris was better than the original game due to its simplicity and ease of play. 
   Ebert's comments were included in the game's DVD commentary, released on March 22, 2010. 
   It is unclear if any of the video commentary was taken from the DVD
'''

```
The arguments are as follows:
* `--input_ids`: The ids of the prefix sequence.
* `--beam_width`: k in the contrastive search, which is typically set within the range of [3,10].
* `--alpha`: alpha in the contrastive search, which is typically set within the range of [0.5,0.8].
* `--decoding_len`: Number of tokens to generate.

**[Note]** We provide two implementations of contrastive search: (1) fast_contrastive_search and (2) slow_contrastive_search. These two implementations produce the same result, but the fast version is properly optimized and is much faster than the slow version. On the other hand, the implementation details of the slow version is more straightforward. We recommend you to rewrite the slow version first if you want to adapt contrastive search to your specific research task.

<span id='diverse_contrastive_search'/>

##### 4.2. Diverse Contrastive Search:
We also provide a stochastic version of contrastive search which can generate diverse results by combining nucleus sampling and contrastive search. More details can be found in Appendix G of the [paper]().
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
* `--sample_step`: The number of tokens sampled with nucleus sampling at the start of the generation process.
* `--nucleus_p`: The probability p in nucleus sampling.
* `--beam_width`: k in the contrastive search, which is typically set within the range of [3,10].
* `--alpha`: alpha in the contrastive search, which is typically set within the range of [0.5,0.8].
* `--decoding_len`: The total number of tokens to generate. It equals to the number of sampled tokens + the number of tokens generated by contrastive search.

<span id='nucleus_sampling'/>

##### 4.3. Nucleus Sampling:
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

##### 4.4. Greedy Search:
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

##### 4.5. Beam Search:
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


****

<span id='evaluation'/>

#### 5. Evaluate the Generated Text

To automatically evaluate the generated results from the language model, please follow the instructions as described [[here]](https://github.com/yxuansu/SimCTG/blob/main/simctg/README.md#documentation-of-the-simctg-library).

****

<span id='visualize_token_similarity_matrix'/>

#### 6. Visualize the Token Similarity Matrix
Here, we show how to reproduce the token similarity matrix visualization (Figure 6 of the paper).
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

# save the visualization result
model.save_token_similarity_map(output, save_name='token_similarity_matrix_visualization.png')
```
The arguments are as follows:
* `--output`: The ids of decoded result.
* `--save_name`: The saved name of the visualization result.

