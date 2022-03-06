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
logging.getLogger('transformers').disabled = True

def parse_config():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument("--model_name", type=str, default='gpt2')
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--dev_path", type=str)
    parser.add_argument("--max_len", type=int, help="length of each sequence example")
    # mini-batch training configuration
    parser.add_argument("--number_of_gpu", type=int, help="Number of available GPUs.")  
    parser.add_argument("--batch_size_per_gpu", type=int, help='batch size for each gpu.') 
    parser.add_argument("--gradient_accumulation_steps", type=int, help="gradient accumulation step.")
    parser.add_argument("--effective_batch_size", type=int, 
        help="effective_bsz = batch_size_per_gpu x number_of_gpu x gradient_accumulation_steps")
    # pre-training configuration
    parser.add_argument("--total_steps", type=int, 
        help="total effective training steps for pre-training stage")
    parser.add_argument("--print_every", type=int, 
        help="how many update steps to print one intermediate result in pre-training stage")
    parser.add_argument("--save_every", type=int, 
        help="how many update steps to save one model in pre-training stage")
    # learning configuration
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--margin", type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--save_path_prefix", type=str, help="directory to save the model parameters.")
    return parser.parse_args()

def evaluate_dev_set_ppl(data, model, cuda_available, device):
    dev_batch_inputs, dev_batch_labels = data.dev_inputs, data.dev_labels
    eval_step = len(dev_batch_labels)
    val_loss, token_sum = 0., 0.
    model.eval()
    with torch.no_grad():
        p = progressbar.ProgressBar(eval_step)
        p.start()
        for idx in range(eval_step):
            p.update(idx)
            batch_input_tensor = dev_batch_inputs[idx]
            batch_labels = dev_batch_labels[idx]
            if cuda_available:
                batch_input_tensor = batch_input_tensor.cuda(device)
                batch_labels = batch_labels.cuda(device)
            if cuda_available and torch.cuda.device_count() > 1: # multi-gpu training
                one_val_loss, one_val_token_sum = model.module.eval_loss(batch_input_tensor, batch_labels)
            else:
                one_val_loss, one_val_token_sum = model.eval_loss(batch_input_tensor, batch_labels)
            one_val_loss = torch.sum(one_val_loss)
            one_val_token_sum = torch.sum(one_val_token_sum)
            val_loss += one_val_loss.item()
            token_sum += one_val_token_sum.item()
        p.finish()
    model.train()
    val_loss = val_loss / token_sum
    val_ppl = np.exp(val_loss)
    return val_ppl

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
            print ('Using single GPU training.')
    else:
        pass
    args = parse_config()
    device = torch.device('cuda')

    batch_size_per_gpu, gradient_accumulation_steps, number_of_gpu, effective_batch_size = \
    args.batch_size_per_gpu, args.gradient_accumulation_steps, args.number_of_gpu, args.effective_batch_size
    assert effective_batch_size == batch_size_per_gpu * gradient_accumulation_steps * number_of_gpu
    model_name = args.model_name

    # load dataset
    print ('Loading dataset...')
    from dataclass import *
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = Data(args.train_path, args.dev_path, tokenizer, batch_size_per_gpu, 
        number_of_gpu, args.max_len)
    print ('Dataset loaded.')

    print ('Initializing model...')
    from simctg import SimCTG
    model = SimCTG(model_name, data.pad_token)
    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model) # multi-gpu training
        else:
            pass
        model = model.to(device)
    else:
        pass
    print ('Model loaded') 

    total_steps, print_every, save_every = args.total_steps, args.print_every, args.save_every
    warmup_steps = int(0.1 * total_steps) # 10% of training steps are used for warmup
    print ('total training steps is {}, warmup steps is {}'.format(total_steps, warmup_steps))
    from transformers.optimization import AdamW, get_linear_schedule_with_warmup
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    optimizer.zero_grad()

    effective_batch_acm = 0
    all_batch_step = 1
    print_valid, save_valid = False, False
    train_loss, train_cl_loss, min_val_ppl = 0., 0., 1e10

    print ('--------------------------------------------------------------------------')
    print ('Start Training:')
    model.train()
    number_of_saves = 0 # keep track of the number of performed save steps
    while effective_batch_acm < total_steps:
        for train_batch_inputs, train_batch_labels in data:
            if effective_batch_acm >= total_steps:
                break # stop training

            all_batch_step += 1
            if cuda_available:
                train_batch_inputs = train_batch_inputs.cuda(device)
                train_batch_labels = train_batch_labels.cuda(device)
            mle_loss, cl_loss = model(train_batch_inputs, train_batch_labels, args.margin)
            loss = mle_loss + cl_loss
            loss = loss.mean()
            loss.backward()
            train_loss += mle_loss.mean().item()
            train_cl_loss += cl_loss.mean().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # parameter update
            if all_batch_step % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                effective_batch_acm += 1
                print_valid, save_valid = True, True

            # print intermediate result
            if effective_batch_acm % print_every == 0 and print_valid:
                denominator = (effective_batch_acm - (number_of_saves * save_every)) * gradient_accumulation_steps
                one_train_loss = train_loss / denominator
                one_train_cl_loss = train_cl_loss / denominator
                print ('At training steps {}, training MLE loss is {}, train CL loss is {}'.format(effective_batch_acm, 
                    one_train_loss, one_train_cl_loss))
                print_valid = False

            # saving result
            if effective_batch_acm % save_every == 0 and save_valid:
                number_of_saves += 1

                save_valid = False
                one_train_loss = train_loss / (save_every * gradient_accumulation_steps)
                one_train_cl_loss = train_cl_loss / (save_every * gradient_accumulation_steps)

                model.eval()
                one_val_ppl = evaluate_dev_set_ppl(data, model, cuda_available, device)
                model.train()

                print ('At training steps {}, training MLE loss is {}, train CL loss is {}, validation ppl is {}'.format(effective_batch_acm, 
                    one_train_loss, one_train_cl_loss, one_val_ppl))

                train_loss, train_cl_loss = 0., 0.
                if one_val_ppl < min_val_ppl:
                    ckpt_save_path = args.save_path_prefix
                    min_val_ppl = min(min_val_ppl, one_val_ppl)
                    print ('Saving model...')
                    one_val_ppl = round(one_val_ppl, 3)
                    save_name = 'training_step_{}_train_mle_loss_{}_train_cl_loss_{}_dev_ppl_{}'.format(effective_batch_acm,
                    round(one_train_loss,3), round(one_train_cl_loss,3), one_val_ppl)
                    model_save_path = ckpt_save_path + '/' + save_name
                    import os
                    if os.path.exists(model_save_path):
                        pass
                    else: # recursively construct directory
                        os.makedirs(model_save_path, exist_ok=True)
                    if cuda_available and torch.cuda.device_count() > 1:
                        model.module.save_model(model_save_path)
                    else:
                        model.save_model(model_save_path)
                    print ('Model Saved!')

                    # --------------------------------------------------------------------------------------------- #
                    # removing extra checkpoints...
                    import os
                    from operator import itemgetter
                    fileData = {}
                    test_output_dir = ckpt_save_path
                    for fname in os.listdir(test_output_dir):
                        if fname.startswith('training_step'):
                            fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
                        else:
                            pass
                    sortedFiles = sorted(fileData.items(), key=itemgetter(1))

                    max_save_num = 1
                    if len(sortedFiles) < max_save_num:
                        pass
                    else:
                        delete = len(sortedFiles) - max_save_num
                        for x in range(0, delete):
                            one_folder_name = test_output_dir + '/' + sortedFiles[x][0]
                            os.system('rm -r ' + one_folder_name)
                    print ('-----------------------------------')
                    # --------------------------------------------------------------------------------------------- #
    print ('Training Completed!')
    print ('--------------------------------------------------------------------------')
