import torch
import random
import logging
logging.getLogger('transformers').disabled = True

import random
import torch
import numpy as np
import progressbar
from torch.nn.utils import rnn

#BUFFER_SIZE = 109600000 # used for pre-training
BUFFER_SIZE = 40960000
class Data:
    def __init__(self, train_path, dev_path, tokenizer, bsz_per_gpu, num_of_gpu, max_len):
        self.bsz_per_gpu, self.num_of_gpu = bsz_per_gpu, num_of_gpu
        self.bsz_one_step = self.bsz_per_gpu * self.num_of_gpu
        self.max_len = max_len
        self.epoch_id = 0
        self.tokenizer = tokenizer
        # add pad token
        # --------------------------------------------------------------------- #
        self.pad_token = '<_PAD_>'
        print ('Original vocabulary size is {}'.format(len(self.tokenizer)))
        self.tokenizer.add_tokens([self.pad_token])
        print ('Vocabulary size after extension is {}'.format(len(self.tokenizer)))
        assert len(self.tokenizer.convert_tokens_to_ids([self.pad_token])) == 1
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids([self.pad_token])[0]
        # --------------------------------------------------------------------- #
        print ('Loading dev data...')
        self.dev_inputs, self.dev_labels = self.load_dev_set(dev_path)
        print ('Dev data loaded.')

        self.train_path, self.dev_path = train_path, dev_path
        self.stream = open(self.train_path, encoding='utf8')

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

    def parse_batch_text(self, batch_text_list):
        batch_id_list = []
        for text in batch_text_list:
            tokens = self.tokenizer.tokenize(text, max_length=self.max_len, truncation=True)
            if len(tokens) <= 1:
                continue
            tokens = tokens[:self.max_len]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            batch_id_list.append(token_ids)
        batch_input_tensor, batch_output_tensor = self.parse_batch(batch_id_list)
        assert batch_input_tensor.size() == batch_output_tensor.size()
        return batch_input_tensor, batch_output_tensor

    def load_dev_set(self, dev_path):
        text_list = []
        with open(dev_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                item_list = l.strip('\n').split('\t')
                assert len(item_list) == 2
                text = item_list[0].strip() + ' ' + self.tokenizer.eos_token + ' ' + item_list[1].strip()
                text = ' '.join(text.split()).strip()
                text_list.append(text)

        dev_inputs, dev_labels = [], []
        batch_num = len(text_list) // self.bsz_one_step
        s_idx, e_idx = 0, self.bsz_one_step
        for _ in range(batch_num):
            one_batch_text_list = text_list[s_idx:e_idx]
            batch_input_tensor, batch_output_tensor = self.parse_batch_text(one_batch_text_list)
            if len(batch_input_tensor) == 0: # ignore empty batch
                continue
            dev_inputs.append(batch_input_tensor)
            dev_labels.append(batch_output_tensor)
            s_idx += self.bsz_one_step
            e_idx += self.bsz_one_step
        print ('Number of dev batches is {}'.format(len(dev_inputs)))
        return dev_inputs, dev_labels

    def __iter__(self):
        lines = self.stream.readlines(BUFFER_SIZE)

        if not lines:
            print ('----------------------------------------')
            self.epoch_id += 1
            print (self.epoch_id)
            self.stream.close()
            self.stream = open(self.train_path, encoding='utf8')
            lines = self.stream.readlines(BUFFER_SIZE)

        text_list = []
        for l in lines:
            item_list = l.strip('\n').split('\t')
            assert len(item_list) == 2
            text = item_list[0].strip() + ' ' + self.tokenizer.eos_token + ' ' + item_list[1].strip()
            text = ' '.join(text.split()).strip()
            text_list.append(text)

        # shuffle the document
        random.shuffle(text_list)
        # create training data
        batch_num = len(text_list) // self.bsz_one_step
        assert batch_num > 0
        idx = 0
        s_idx, e_idx = 0, self.bsz_one_step
        while idx < batch_num:
            one_batch_text_list = []
            for text in text_list[s_idx:e_idx]:
                one_batch_text_list.append(text)
            s_idx += self.bsz_one_step
            e_idx += self.bsz_one_step
            idx += 1
            yield self.parse_batch_text(one_batch_text_list)

