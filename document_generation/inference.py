# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse, os
import random
import numpy as np
import time
import logging
import progressbar

import logging
logging.getLogger('transformers.generation_utils').disabled = True

def inference_one_instance(args, data, model, index, k, alpha, cuda_available, device):
    one_res_dict = {}
    one_res_dict['prefix_text'] = data.prefix_text_list[index]
    one_res_dict['reference_text'] = data.reference_text_list[index]
    one_res_dict['reference_continuation_text'] = data.reference_continuation_text_list[index]

    generated_dict = {}

    input_ids = data.prefix_token_id_list[index]
    input_tensor = torch.LongTensor(input_ids).view(1,-1)
    if cuda_available:
        input_tensor = input_tensor.cuda(device)

    num_per_instance = args.num_per_instance
    for idx in range(num_per_instance):
        output = model.fast_contrastive_search(input_tensor, k, alpha, args.decoding_len)
        output_text = data.tokenizer.decode(output)
        output_continuation = data.tokenizer.decode(output[args.prefix_len:])
        generated_dict[idx] = {'full_text': output_text, 'continuation':output_continuation}
    one_res_dict['generated_result'] = generated_dict
    return one_res_dict

def inference_one_file(args, data, model, cuda_available, device):
    print ('----------------------------------------------------------------')
    print ('Start inference...')
    save_path = args.save_path
    test_num = len(data.prefix_token_id_list)
    result_list = []
    p = progressbar.ProgressBar(test_num)
    p.start()
    with torch.no_grad():
        for index in range(test_num):
            p.update(index)
            one_res_dict = inference_one_instance(args, data, model, index, args.k, 
                args.alpha, cuda_available, device)
            result_list.append(one_res_dict)
        p.finish()
    import json
    with open(save_path, 'w') as outfile:
        json.dump(result_list, outfile, indent=4)
    print ('Inference completed.')

def parse_config():
    parser = argparse.ArgumentParser()
    # model and data configuration
    parser.add_argument("--ckpt_path", type=str, help="path of the pre-trained checkpoint")
    parser.add_argument("--dev_path", type=str)
    parser.add_argument("--test_path", type=str)
    # evaluation configuration
    parser.add_argument("--prefix_len", type=int, default=32)
    parser.add_argument("--decoding_len", type=int, default=128)
    parser.add_argument("--num_per_instance", type=int, help="how many samples to generate per instance.")
    # save configuration
    parser.add_argument("--k", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--save_path", type=str)
    return parser.parse_args()

import argparse
if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print ('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            print ('Using single GPU.')
    else:
        pass
    args = parse_config()
    device = torch.device('cuda')

    print ('Loading data...')
    from inference_dataclass import Data
    data = Data(args.ckpt_path, args.dev_path, args.test_path, args.prefix_len, args.decoding_len)
    print ('Data loaded.')

    print ('Loading pre-trained model...')
    from simctg import SimCTG
    model = SimCTG(args.ckpt_path, data.pad_token_id)
    if cuda_available:
        model = model.to(device)
    model.eval()
    print ('Model loaded') 

    with torch.no_grad():
        inference_one_file(args, data, model, cuda_available, device)
