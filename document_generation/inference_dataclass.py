import random
import torch
import numpy as np
import progressbar
from torch.nn.utils import rnn

class Data:
    def __init__(self, model_name, dev_path, test_path, prefix_len, decoding_len):
        '''
            dev_path, test_path: data path to validate the result
            prefix_len: length of the human-written prefix
            decoding_len: length of generated text continuation
        '''
        from transformers import GPT2TokenizerFast
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.bos_token])[0]
        print ('padding token is {}, padding token id {}'.format(self.tokenizer.bos_token, self.pad_token_id))
        self.prefix_len, self.decoding_len = prefix_len, decoding_len
        self.min_len = self.prefix_len + self.decoding_len

        dev_prefix_token_id_list, dev_prefix_text_list, dev_reference_text_list, \
        dev_reference_continuation_text_list = self.process_one_file(dev_path)

        test_prefix_token_id_list, test_prefix_text_list, test_reference_text_list, \
        test_reference_continuation_text_list = self.process_one_file(test_path)

        # combine data
        self.prefix_token_id_list = dev_prefix_token_id_list + test_prefix_token_id_list
        self.prefix_text_list = dev_prefix_text_list + test_prefix_text_list
        self.reference_text_list = dev_reference_text_list + test_reference_text_list
        self.reference_continuation_text_list = dev_reference_continuation_text_list + \
        test_reference_continuation_text_list
        print ('Evaluation number is {}'.format(len(self.prefix_token_id_list)))

    def process_one_file(self, path):
        print ('Processing {}'.format(path))
        prefix_token_id_list, prefix_text_list, reference_text_list, \
        reference_continuation_text_list = [], [], [], []

        with open(path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
        n = len(lines)
        print (n)
        p = progressbar.ProgressBar(n)
        p.start()
        for i in range(n):
            p.update(i)
            text = lines[i].strip('\n')
            self.process_one_text(text, prefix_token_id_list, prefix_text_list, reference_text_list,
                                  reference_continuation_text_list)
        p.finish()
        print ('{} processed!'.format(path))
        return prefix_token_id_list, prefix_text_list, reference_text_list, reference_continuation_text_list

    def process_one_text(self, text, prefix_token_id_list, prefix_text_list, reference_text_list, \
        reference_continuation_text_list):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) < self.min_len:
            return

        token_id_list = self.tokenizer.convert_tokens_to_ids(tokens)
        prefix_id_list = token_id_list[:self.prefix_len]
        prefix_token_id_list.append(prefix_id_list)
        prefix_text = self.tokenizer.decode(prefix_id_list)
        prefix_text_list.append(prefix_text)
        reference_text_list.append(text)
        reference_continuation_text = self.tokenizer.decode(token_id_list[self.prefix_len:])
        reference_continuation_text_list.append(reference_continuation_text)
        return 
