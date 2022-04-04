## Open-Ended Story Generation

**[Note]** In this directory, we only show how to adapt our approach to open-ended story generation. The performance of the model is not rigoriously tested with human evaluation. But we hope it can help people who are interested in this task.

****
### Catalogue:
* <a href='#writingprompts'>1. Open-Ended Story Generation on WritingPrompts Benchmark</a>
    * <a href='#data_preparation'>1.1 Data Preparation</a>
    * <a href='#train_simctg'>1.2 Train SimCTG</a>
    * <a href='#generate_results'>1.3 Generate Result</a>
* <a href='#rocstories'>2. Open-Ended Story Generation on ROCStories Benchmark</a>
    * <a href='#roc_data_preparation'>2.1 Data Preparation</a>
    * <a href='#roc_train_simctg'>2.2 Train SimCTG</a>
    * <a href='#roc_generate_results'>2.3 Generate Result</a>
    
****

<span id='writingprompts'/>

### 1. Open-Ended Story Generation on WritingPrompts Benchmark:
This benchmark is proposed by [Fan et al. (2018)](https://arxiv.org/abs/1805.04833).

<span id='data_preparation'/>

#### 1.1 Data Preparation:
To download the WritingPrompts data, please follow the instructions [[here]](https://github.com/yxuansu/SimCTG/tree/main/data).

> **** The dataset contains the following three files:

    .
    ├── WritePrompt                     
        ├── writeprompt_train.txt   # Training Set
        ├── writeprompt_dev.txt     # Validation Set
        └── writeprompt_test.txt    # Test Set

**Data Format**: In the files, each line is formatted as prompt + '\t' + story.


<span id='train_simctg'/>

#### 1.2 Train SimCTG:
To train a SimCTG model on WritePrompt, please run the following commands:
```yaml
chmod +x ./train.sh
./train.sh
```
The arguments are as follows:
* `--model_name`: The name of huggingface pre-trained gpt model (e.g. gpt2, gpt-large).
* `--train_path`: The file path of training set.
* `--dev_path`: The file path of validation set.
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


<span id='generate_results'/>

#### 1.3 Generate Result:
```python
# define function that loads test prompts
def load_prompt(in_f):
    prompt_list = []
    with open(in_f, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        for l in lines:
            item_list = l.strip('\n').split('\t')
            prompt = item_list[0].strip()
            prompt_list.append(prompt)
    return prompt_list

# load test set prompts
in_f = r'../data/WritePrompt/writeprompt_test.txt'
prompt_list = load_prompt(in_f)
```

```python
import torch
import sys
from simctg import SimCTG
# load model
model_path = r'cambridgeltl/simctg_writingprompts'
pad_token = '<_PAD_>'
model = SimCTG(model_path, pad_token)
model.eval()

# ---------- example 1 ---------- #
index = 3
prompt = prompt_list[index] + ' ' + model.tokenizer.eos_token
print ('prompt is:')
print (prompt)
tokens = model.tokenizer.tokenize(prompt)
input_ids = model.tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor(input_ids).view(1,-1)
'''
    prompt is:
    [ WP ] A kid doodling in a math class accidentally creates the world 's first
    functional magic circle in centuries . <|endoftext|>
'''

beam_width, alpha, decoding_len = 5, 0.6, 200
output = model.fast_contrastive_search(input_ids, beam_width, alpha, decoding_len)
generated_story = model.tokenizer.decode(output).split(model.tokenizer.eos_token)[1].strip()
print ('generated story is:')
print (generated_story)
'''
    generated story is:
    I looked at the circle, it wasn't there. I couldn't see it, and my eyes were watering 
    from the rain that had fallen over the school, the wind howling through the windows 
    and making a wispy noise as it passed through the air. `` What is it? '' I asked, trying 
    to find the source of the noise. `` It's a circle, '' the teacher said in a voice that 
    sounded like it was from an old TV show or something like that. `` You can't make it out
    of there. '' I looked around the room, there was no one there. It was as if I was in a 
    dream, but no one seemed to notice me. Then I saw a flash of light, and the circle appeared
    in front of me. I turned around to see what was going on, I had never seen anything like
    it before in my life. I ran up to the teacher and asked, `` Are you sure this is real?
'''
```
The arguments are as follows:
* `--input_ids`: The ids of the prefix sequence.
* `--beam_width`: k in the contrastive search, which is typically set within the range of [3,10].
* `--alpha`: alpha in the contrastive search, which is typically set within the range of [0.5,0.8].
* `--decoding_len`: Number of tokens to generate.


```python
# ---------- example 2 ---------- #
index = 32
prompt = prompt_list[index] + ' ' + model.tokenizer.eos_token
print ('prompt is:')
print (prompt)
tokens = model.tokenizer.tokenize(prompt)
input_ids = model.tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor(input_ids).view(1,-1)
'''
    prompt is:
    [ WP ] On the day of their death , every human gets to have a vision of meeting the man
    they could have become . <|endoftext|>
'''

beam_width, alpha, decoding_len = 5, 0.6, 200
output = model.fast_contrastive_search(input_ids, beam_width, alpha, decoding_len)
generated_story = model.tokenizer.decode(output).split(model.tokenizer.eos_token)[1].strip()
print ('generated story is:')
print (generated_story)
'''
    generated story is:
    It wasn't long after the event that I realized it was a dream. My vision was a blur, like
    I was floating in a sea of white clouds that were not my own at all. I looked around me,
    but nothing was there. There was no light, no sound, just silence. Then, as if on cue, the
    clouds parted and I was standing on a white marble floor. A man in a lab coat, glasses, and
    a pair of black sunglasses stared at me with a blank expression. `` What are you doing here? ''
    I asked, trying to find the words that had made me think of this man. He didn't answer, instead
    focusing on his work in the lab. `` You're going to die in an instant, '' he said in a monotone
    voice. `` But I'm not going to die. '' I was confused for a moment, but then I heard a voice
    from the other side of the room. `` Do n
'''
```

****

<span id='rocstories'/>

### 2. Open-Ended Story Generation on ROCStories Benchmark:
This benchmark is proposed by [Mostafazadeh et al. (2016)](https://aclanthology.org/N16-1098/).

<span id='roc_data_preparation'/>

#### 2.1 Data Preparation:
To download the ROCStories data, please follow the instructions [[here]](https://github.com/yxuansu/SimCTG/tree/main/data).

> **** The dataset contains the following files:

    .
    ├── ROCStories             
        ├── rocstories_train.txt   # Training Set
        └── rocstories_test.txt    # Test Set

**Data Format**: In the files, each line is formatted as prompt + '\t' + story.

<span id='roc_train_simctg'/>

#### 2.2 Train SimCTG:
To train a SimCTG model on ROCStories, please run the following commands:
```yaml
chmod +x ./train_rocstories.sh
./train_rocstories.sh
```
The arguments have been described in section 1.2.

<span id='roc_generate_results'/>

#### 2.3 Generate Result:

```python
import torch
import sys
from simctg import SimCTG
# load model
model_path = r'cambridgeltl/simctg_rocstories'
pad_token = '<_PAD_>'
model = SimCTG(model_path, pad_token)
model.eval()

prompt = r"Accident in the Lab <|endoftext|>"
tokens = model.tokenizer.tokenize(prompt)
input_ids = model.tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor(input_ids).view(1,-1)

beam_width, alpha, decoding_len = 5, 0.65, 64
output = model.fast_contrastive_search(input_ids, beam_width, alpha, decoding_len)
generated_story = model.tokenizer.decode(output).split(model.tokenizer.eos_token)[1].strip()
print ('generated story is:')
print (generated_story)
'''
    generated story is:
    Tom went to work one day. He noticed a lab accident in the lab. Tom was worried about his safety at 
    work. Unfortunately the accident didn't go well. Tom wound up leaving early to get back on the job.
'''
```
