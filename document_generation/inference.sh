CUDA_VISIBLE_DEVICES=0 python inference.py\
    --ckpt_path cambridgeltl/simctg_wikitext103\
    --dev_path ../data/language_modelling/wikitext103/wikitext103_raw_v1_validation.txt\
    --test_path ../data/language_modelling/wikitext103/wikitext103_raw_v1_test.txt\
    --prefix_len 32\
    --decoding_len 128\
    --num_per_instance 1\
    --k 8\
    --alpha 0.6\
    --save_path simctg_contrastive.json
