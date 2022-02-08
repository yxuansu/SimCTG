import torch
import random
import logging
logging.getLogger('transformers').disabled = True

BUFFER_SIZE = 109600000 # used for pre-training
#BUFFER_SIZE = 10960000
#BUFFER_SIZE = 1096000 # used for debugging purpose
class EnglishPretrainCorpus:
    def __init__(self, train_path, dev_path, tokenizer, bsz_per_gpu, num_of_gpu, seqlen):
        '''
            train_path: large size training data
            dev_path: smaller size validation data
                The format of train and dev data is: each line is a wiki document

            tokenizer: e.g. GPT tokenizer
            bsz_per_gpu: at each forward step, how many examples (i.e. sequences) are assigned to a single gpu
            num_of_gpu: number of available gpus during training
            seqlen: the length of each sequence (e.g. 512)
        '''
        self.tokenizer = tokenizer
        self.train_path, self.dev_path = train_path, dev_path
        self.stream = open(self.train_path, encoding='utf8')
        self.bsz_per_gpu, self.num_of_gpu = bsz_per_gpu, num_of_gpu
        self.bsz_one_step = self.bsz_per_gpu * self.num_of_gpu
        self.epoch_id = 0
        self.eos_token = self.tokenizer.eos_token
        self.seqlen = seqlen
        self.block_size = self.seqlen + 1 # input: [:-1]; label:[1:]
        print ('Loading dev data...')
        self.dev_inputs, self.dev_labels = self.load_dev_set(dev_path)
        print ('Dev data loaded.')

    def load_dev_set(self, dev_path):
        text_list = []
        with open(dev_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                text_list.append(l.strip('\n').strip())
        single_text = ''
        for text in text_list:
            single_text += text + self.eos_token
        str_ids = self.tokenizer.encode(single_text)
        str_len = len(str_ids)
        example_num = str_len // self.block_size
        buffer_examples = []
        start_idx, end_idx = 0, self.block_size
        for _ in range(example_num):
            one_buffer = str_ids[start_idx:end_idx]
            assert len(one_buffer) == self.seqlen + 1
            buffer_examples.append(one_buffer)
            start_idx += self.block_size
            end_idx += self.block_size

        dev_inputs, dev_labels = [], []
        bsz_one_step = int(self.bsz_one_step/2)
        s_idx, e_idx = 0, bsz_one_step
        batch_num = len(buffer_examples) // bsz_one_step
        for _ in range(batch_num):
            one_dev_input, one_dev_label = [], []
            for exp in buffer_examples[s_idx:e_idx]:
                one_dev_input.append(exp[:-1])
                one_dev_label.append(exp[1:])
            dev_inputs.append(torch.LongTensor(one_dev_input))
            dev_labels.append(torch.LongTensor(one_dev_label))
            s_idx += bsz_one_step
            e_idx += bsz_one_step
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
            # each line of the lines is a single document
            text_list.append(l.strip('\n').strip())

        # shuffle the document
        random.shuffle(text_list)
        # create single long text from the buffered lines
        single_text = ''
        for text in text_list:
            single_text += text + self.eos_token
        # tokenize the long text
        str_ids = self.tokenizer.encode(single_text)
        # split the long string ids into segments with predefined length
        str_len = len(str_ids)
        example_num = str_len // self.block_size
        buffer_examples = []
        start_idx, end_idx = 0, self.block_size
        for _ in range(example_num):
            one_buffer = str_ids[start_idx:end_idx]
            assert len(one_buffer) == self.seqlen + 1
            buffer_examples.append(one_buffer)
            start_idx += self.block_size
            end_idx += self.block_size
        random.shuffle(buffer_examples)

        # fetch batch data from buffer_examples
        batch_num = example_num // self.bsz_one_step
        assert batch_num > 0
        idx = 0
        s_idx, e_idx = 0, self.bsz_one_step
        while idx < batch_num:
            inputs, labels = [], []
            for one_example_id in buffer_examples[s_idx:e_idx]:
                inputs.append(one_example_id[:-1])
                labels.append(one_example_id[1:])
            assert len(inputs) == self.bsz_one_step
            s_idx += self.bsz_one_step
            e_idx += self.bsz_one_step
            idx += 1
            yield torch.LongTensor(inputs), torch.LongTensor(labels)
