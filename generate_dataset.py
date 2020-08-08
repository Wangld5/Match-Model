import sys
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader
import csv
from tqdm import tqdm
import re
import os
import random
import pandas as pd
import argparse
import json
import numpy as np


def _format_sentence(sent):
    s = sent.strip().split()
    s = ['<unk>' if w == '<unknown>' else w for w in s]
    s = ' '.join(s)
    s = s.lower()
    s = re.sub(r"[^0-9a-z]", ' ', s)
    s = re.sub(r"\s+", ' ', s)
    s = s.strip().split()
    s = ' '.join(s)
    s = s.split()
    if len(s) == 0:
        s.append('unk')
    return ' '.join(s)


def _format_sentence2(sent):
    pass


def read_file(path):
    context = open(path, 'r').readlines()
    lines = []
    for line in context:
        lines.append(line.strip())
    return lines


def read_template_index(template_file, k, keyword):
    if keyword == 'train':
        start = 0
    else:
        start = 0
    template_origin = open(template_file, 'r').readlines()
    template_indexs = []
    for template in template_origin:
        template_filter = [index.split('.txt')[0] for index in template.strip().split()]
        template_indexs.append(template_filter[:k])
    return template_indexs


def generate_dataset_index(article, template_file, k, save_path):
    template_origin = open(template_file, 'r').readlines()
    dataset_index = []
    article_sample = range(0, len(article))
    for i in tqdm(article_sample):
        template_indexs = [index.split('.txt')[0] for index in template_origin[i].strip().split()][:k]
        dataset_index.append({'art_idx': i, 'tp_idx': template_indexs})
    json.dump(dataset_index, open(save_path, 'w'))


def generate_dataset(dataset_index, template_corpus, article, title, save_path):
    templates_sample = [template_corpus[i[2]].strip() for i in dataset_index]
    titles_sample = [title[i[1]].strip() for i in dataset_index]
    articles_sample = [article[i[0]].strip() for i in dataset_index]
    print(templates_sample[1])
    print(titles_sample[1])
    print(articles_sample[1])
    print(templates_sample[2], titles_sample[2], articles_sample[2])
    print(len(templates_sample), len(titles_sample), len(articles_sample))

    save_files = pd.DataFrame({'article': articles_sample, 'truth': titles_sample, 'template': templates_sample})
    save_files.to_csv(save_path, index=False, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-template_index', type=str, required=True)
    parser.add_argument('-article', type=str, required=True)
    parser.add_argument('-title', type=str)
    parser.add_argument('-template_corpus', type=str)
    parser.add_argument('-k', type=int, default=15)
    parser.add_argument('-sample_num', type=int)
    parser.add_argument('-save_path', type=str, required=True)
    parser.add_argument('-keyword', type=str)
    args = parser.parse_args()

    # template_corpus = read_file(args.template_corpus)
    articles = read_file(args.article)
    # titles = read_file(args.title)
    # template_index = read_template_index(args.template_index, args.k, args.keyword)
    generate_dataset_index(articles, args.template_index, args.k, args.save_path)
    # generate_dataset(dataset_index, template_corpus, articles, titles, args.save_path)
