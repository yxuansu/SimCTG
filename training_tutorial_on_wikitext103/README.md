### Reproduction of our experiments on Wikitext-103 using the released package.

(1) First, please prepare the dataset as instructed [[here]](https://github.com/yxuansu/SimCTG/tree/main/data#1-download-wikitext-103-dataset).

(2) To train a SimCTG model on Wikitext-103, please run the following commands:
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

(3) For more details of the experiment setup, please refer to [[here]](https://github.com/yxuansu/SimCTG/tree/main/document_generation).
