import torch
import os
import sys
import numpy as np
import argparse
from training import train, dev
from preprocess import preproc


parser = argparse.ArgumentParser()
parser.add_argument('-train_max_len', type=int, default=50, help="limited length of train data")
parser.add_argument('-dev_max_len', type=int, default=100, help="limited length of dev data")
parser.add_argument('-template_max_len', type=int, default=30, help="limited length of template token")
parser.add_argument('-title_max_len', type=int, default=30, help='limited length of title token')
parser.add_argument('-train_article', type=str, help='train article file path')
parser.add_argument('-train_title', type=str, help='train title file path')
parser.add_argument('-train_dataset_index', type=str, help="train data index file path")
parser.add_argument('-dev_article', type=str, help='dev article file path')
parser.add_argument('-dev_title', type=str, help='dev title file path')
parser.add_argument('-dev_dataset_index', type=str, help='dev data index file path')
parser.add_argument('-test_article', type=str, help='test article file path')
parser.add_argument('-test_title', type=str, help='test title file path')
parser.add_argument('-test_dataset_index', type=str, help='test data index file path')
parser.add_argument('-fasttext_file', type=str, default=None)
parser.add_argument('-fasttext', type=bool, default=False)
parser.add_argument('-glove_word_file', type=str, default=None)
parser.add_argument('-train_token_file', type=str, help='path to save train token')
parser.add_argument('-dev_token_file', type=str, help='path to save dev token')
parser.add_argument('-test_token_file', type=str, help='path to save test token')
parser.add_argument('-word_emb_file', type=str, help='path to save word embedding')
parser.add_argument('-word_len', type=int, default=0, help='word length')
parser.add_argument('-train_log', type=str, default=None, help='training log')
parser.add_argument('-batch_size', type=int, default=48, help='training batch size')
parser.add_argument('-val_batch_size', type=int, default=60, help='valid batch size')
parser.add_argument('-word_dim', type=int, default=300, help='word dimension')
parser.add_argument('-kernel_size', type=int, default=3, help='CNN kernel size')
parser.add_argument('-encoder_block_num', type=int, default=1, help='encoder block num')
parser.add_argument('-model', type=str, default=None, help='model name for save')
parser.add_argument('-save_dir', type=str, default=None, help='saving path for model')
parser.add_argument('-L2_norm', type=float, help='L2 norm for optimizer', default=3e-6)
parser.add_argument('-learning_rate', type=float, default=1e-2, help='learning rate')
parser.add_argument('-margin', type=float, default=0.3, help='margin for ranking loss')
parser.add_argument('-epochs', type=int, default=10, help='epochs for cycle')
parser.add_argument('-grad_clip', type=float, default=5.0, help='for global gradient clipping')
parser.add_argument('-checkpoint', type=int, default=10000, help='checkpoint for validation')
parser.add_argument('-early_stop', type=int, default=10, help='checkpoint for early stop training')
parser.add_argument('-keyword', type=str, help='keyword for test')
parser.add_argument('-dev_log', type=str, default=None, help='dev log file')
parser.add_argument('-template_save', type=str, help='path to save template')
parser.add_argument('-template_num', type=int, help='number for template for choosing')
parser.add_argument('-train_template', type=str, help='file to save train template')
parser.add_argument('-dev_template', type=str, help='file to save dev template')
parser.add_argument('-test_template', type=str, help='file to save test template')
parser.add_argument('-pred_file', type=str, help='file to save format template')
parser.add_argument('-mode', type=str)
config = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    if config.mode == 'train':
        train(config, device)
    elif config.mode == 'preprocess':
        preproc(config)
    elif config.mode == 'dev':
        dev(config, device)
    else:
        print("unknown mode")
        exit(0)


