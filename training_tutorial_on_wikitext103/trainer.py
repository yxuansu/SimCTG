# coding=utf-8
import sys
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

def eval_model(args, model, data, cuda_available, device):
    dataset_batch_size = args.batch_size_per_gpu * args.number_of_gpu
    eval_step = int(data.test_num / dataset_batch_size) + 1
    val_loss, token_sum = 0., 0.
    model.eval()
    with torch.no_grad():
        p = progressbar.ProgressBar(eval_step)
        p.start()
        for idx in range(eval_step):
            p.update(idx)
            batch_input_tensor, batch_labels, _ = \
            data.get_next_validation_batch(batch_size=dataset_batch_size, mode='test')
            if cuda_available:
                batch_input_tensor = batch_input_tensor.cuda(device)
                batch_labels = batch_labels.cuda(device)
            one_val_loss, one_val_token_sum = model.eval_loss(batch_input_tensor, batch_labels)
            one_val_loss = torch.sum(one_val_loss)
            one_val_token_sum = torch.sum(one_val_token_sum)
            val_loss += one_val_loss.item()
            token_sum += one_val_token_sum.item()
        p.finish()
    model.train()
    val_loss = val_loss / token_sum
    return val_loss

def model_training(args, data, model, vocab_size, total_steps, print_every, save_every, 
    ckpt_save_path, cuda_available, device):
    # initialize loss category
    from simctg.lossfunction import SimCTGLoss
    margin, pad_token_id = args.margin, data.pad_token_id
    simctgloss = SimCTGLoss(margin, vocab_size, pad_token_id)

    import os
    if os.path.exists(ckpt_save_path):
        pass
    else: # recursively construct directory
        os.makedirs(ckpt_save_path, exist_ok=True)
    log_path = ckpt_save_path + '/log.txt'

    max_save_num = 1
    batch_size_per_gpu, gradient_accumulation_steps, number_of_gpu, effective_batch_size = \
    args.batch_size_per_gpu, args.gradient_accumulation_steps, args.number_of_gpu, args.effective_batch_size
    assert effective_batch_size == batch_size_per_gpu * gradient_accumulation_steps * number_of_gpu

    warmup_steps = int(0.1 * total_steps) # 10% of training steps are used for warmup
    print ('total training steps is {}, warmup steps is {}'.format(total_steps, warmup_steps))
    from transformers.optimization import AdamW, get_linear_schedule_with_warmup
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    optimizer.zero_grad()

    effective_batch_acm = 0
    all_batch_step = 1
    print_valid, save_valid = False, False
    train_loss, train_cl_loss, min_val_loss = 0., 0., 1e10
    train_ave_bleu = 0.

    print ('--------------------------------------------------------------------------')
    print ('Start Training:')
    model.train()
    number_of_saves = 0

    while effective_batch_acm < total_steps:
        all_batch_step += 1
        train_batch_input_tensor, train_batch_labels, _ = data.get_next_train_batch(batch_size_per_gpu * number_of_gpu)
        if cuda_available:
            train_batch_input_tensor = train_batch_input_tensor.cuda(device)
            train_batch_labels = train_batch_labels.cuda(device)
        last_hidden_states, logits = model(train_batch_input_tensor, train_batch_labels)
        mle_loss, cl_loss = simctgloss(last_hidden_states, logits, train_batch_input_tensor, train_batch_labels)

        loss = mle_loss + cl_loss
        loss = loss.mean()
        loss.backward()
        train_loss += mle_loss.item()
        train_cl_loss += cl_loss.item()
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
            train_log_text = 'Training:At training steps {}, training MLE loss is {}, train CL loss is {}'.format(effective_batch_acm, 
                one_train_loss, one_train_cl_loss)
            print (train_log_text)
            with open(log_path, 'a', encoding='utf8') as logger:
                logger.writelines(train_log_text + '\n')
            print_valid = False

        # saving result
        if effective_batch_acm % save_every == 0 and save_valid:
            number_of_saves += 1

            save_valid = False
            one_train_loss = train_loss / (save_every * gradient_accumulation_steps)
            one_train_cl_loss = train_cl_loss / (save_every * gradient_accumulation_steps)

            model.eval()
            one_val_loss = eval_model(args, model, data, cuda_available, device)
            one_val_ppl = np.exp(one_val_loss)
            one_val_ppl = round(one_val_ppl, 3)
            model.train()

            valid_log_text = 'Validation:At training steps {}, training MLE loss is {}, train CL loss is {}, validation loss is {}, validation ppl is {}'.format(effective_batch_acm, 
                one_train_loss, one_train_cl_loss, one_val_loss, one_val_ppl)
            print (valid_log_text)
            with open(log_path, 'a', encoding='utf8') as logger:
                logger.writelines(valid_log_text + '\n')

            train_loss, train_cl_loss = 0., 0.

            if one_val_loss < min_val_loss:
                # in finetuning stage, we always save the model
                min_val_loss = min(one_val_loss, min_val_loss)
                print ('Saving model...')
                one_val_ppl = np.exp(one_val_loss)
                one_val_ppl = round(one_val_ppl, 3)
                save_name = 'training_step_{}_train_mle_loss_{}_train_cl_loss_{}_dev_loss_{}_dev_ppl_{}'.format(effective_batch_acm,
                round(one_train_loss,5), round(one_train_cl_loss,5), round(one_val_loss,5), one_val_ppl)

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

                if len(sortedFiles) < max_save_num:
                    pass
                else:
                    delete = len(sortedFiles) - max_save_num
                    for x in range(0, delete):
                        one_folder_name = test_output_dir + '/' + sortedFiles[x][0]
                        os.system('rm -r ' + one_folder_name)
                print ('-----------------------------------')
                # --------------------------------------------------------------------------------------------- #
    return model

