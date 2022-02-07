import random
import torch
import numpy as np
import progressbar
from torch.nn.utils import rnn

class Data:
    def __init__(self, model_name, train_path, dev_path, test_path, min_len, max_len, eos_token, pad_token):
        '''
            model_name: gpt2
            train_path: training data path
            dev_path: validation data path
            test_path: test data path 
            min_len: minimum length for training sequences
            max_len: maximum length for training sequences 
        '''
        #from transformers import GPT2TokenizerFast
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.min_len, self.max_len = min_len, max_len
        if eos_token == 'endoftext':
            print ('Use English GPT')
            self.eos_token = self.tokenizer.eos_token
        else:
            self.eos_token = eos_token
        self.train_token_list, self.train_token_id_list = self.process_one_file(train_path)
        self.dev_token_list, self.dev_token_id_list = self.process_one_file(dev_path)
        self.test_token_list, self.test_token_id_list = self.process_one_file(test_path)
        self.train_num, self.dev_num, self.test_num = len(self.train_token_list), len(self.dev_token_list), \
        len(self.test_token_list)
        print ('train number:{}, dev number:{}, test number:{}'.format(self.train_num, self.dev_num, self.test_num))

        #if len(self.tokenizer.convert_tokens_to_ids([pad_token])) == 1: # pad token is in the vocabulary
        if pad_token in self.tokenizer.vocab:
            print ('PAD token exists.')
        else:
            print ('Add PAD token to the tokenizer.')
            print ('Original vocabulary size is {}'.format(len(self.tokenizer)))
            self.tokenizer.add_tokens([pad_token])
            print ('Vocabulary size after extension is {}'.format(len(self.tokenizer)))
            assert len(self.tokenizer.convert_tokens_to_ids([pad_token])) == 1
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids([pad_token])[0]
        print ('padding token is {}, padding token id {}'.format(pad_token, self.pad_token_id))

        self.train_idx_list = [i for i in range(self.train_num)]
        random.shuffle(self.train_idx_list)
        self.dev_idx_list = [j for j in range(self.dev_num)]
        self.test_idx_list = [j for j in range(self.test_num)]
        self.dev_current_idx, self.test_current_idx = 0, 0


    def process_one_file(self, path):
        print ('Processing {}'.format(path))
        res_token_list, res_token_id_list = [], []
        with open(path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
        n = len(lines)
        p = progressbar.ProgressBar(n)
        p.start()
        for i in range(n):
            p.update(i)
            text = lines[i].strip('\n')
            self.process_one_text(text, res_token_list, res_token_id_list)
        p.finish()
        print ('{} processed!'.format(path))
        return res_token_list, res_token_id_list

    def process_one_text(self, text, res_token_list, res_token_id_list):
        text_list = text.strip('\n').strip('\t').split('\t')
        if len(text_list) <= 1:
            return
        text = ''
        for one_text in text_list:
            text += one_text + self.eos_token
        tokens = self.tokenizer.tokenize(text)[-self.max_len:]
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= self.min_len:
            return
        token_ids = token_ids[-self.max_len:]
        assert len(token_ids) <= self.max_len
        res_token_list.append(tokens)
        res_token_id_list.append(token_ids)
        return

    def pad_batch(self, batch_id_list):
        batch_id_list = [torch.LongTensor(item) for item in batch_id_list]
        batch_tensor = rnn.pad_sequence(batch_id_list, batch_first=True, padding_value=self.pad_token_id)
        batch_mask = torch.ones_like(batch_tensor)
        batch_mask = batch_mask.masked_fill(batch_tensor.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
        return batch_tensor, batch_mask

    def process_output(self, batch_tgt_id_list):
        batch_tgt_id_list = [torch.LongTensor(item) for item in batch_tgt_id_list]
        batch_tgt_tensor, _ = self.pad_batch(batch_tgt_id_list) # padded target sequence
        batch_tgt_input_tensor = batch_tgt_tensor[:, :-1].clone()
        batch_tgt_output_tensor = batch_tgt_tensor[:, 1:].clone()
        return batch_tgt_input_tensor, batch_tgt_output_tensor

    def parse_batch(self, batch_id_list):
        batch_input, batch_labels = self.process_output(batch_id_list)
        batch_labels[batch_labels[:, :] == self.pad_token_id] = -100
        return batch_input, batch_labels

    def get_next_train_batch(self, batch_size):
        batch_idx_list = random.sample(self.train_idx_list, batch_size)
        batch_id_list, batch_token_list = [], []

        for idx in batch_idx_list:
            batch_id_list.append(self.train_token_id_list[idx])
            batch_token_list.append(self.train_token_list[idx])
        batch_input_tensor, batch_labels = self.parse_batch(batch_id_list)
        return batch_input_tensor, batch_labels, batch_token_list

    def get_next_validation_batch(self, batch_size, mode):
        batch_id_list, batch_token_list = [], []
        if mode == 'dev':
            curr_select_idx, instance_num = self.dev_current_idx, self.dev_num
            tgt_token_id_list, tgt_token_list = self.dev_token_id_list, self.dev_token_list
        elif mode == 'test':
            curr_select_idx, instance_num = self.test_current_idx, self.test_num
            tgt_token_id_list, tgt_token_list = self.test_token_id_list, self.test_token_list
        else:
            raise Exception('Wrong Validation Mode!!!')

        if curr_select_idx + batch_size < instance_num:
            for i in range(batch_size):
                curr_idx = curr_select_idx + i
                batch_id_list.append(tgt_token_id_list[curr_idx])
                batch_token_list.append(tgt_token_list[curr_idx])
            if mode == 'dev':
                self.dev_current_idx += batch_size
            else:
                self.test_current_idx += batch_size
        else:
            for i in range(batch_size):
                curr_idx = curr_select_idx + i
                if curr_idx > instance_num - 1: 
                    curr_idx = 0
                    if mode == 'dev':
                        self.dev_current_idx = 0
                    else:
                        self.test_current_idx = 0
                batch_id_list.append(tgt_token_id_list[curr_idx])
                batch_token_list.append(tgt_token_list[curr_idx])
            if mode == 'dev':
                self.dev_current_idx = 0
            else:
                self.test_current_idx = 0
        batch_input_tensor, batch_labels = self.parse_batch(batch_id_list)
        return batch_input_tensor, batch_labels, batch_token_list
