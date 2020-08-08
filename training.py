import os
import numpy as np
import ujson as json
from tqdm import tqdm
import torch
import torch.optim as optim
from model import FastRerank
import torch.nn.functional as F
from utils import get_logger
import time
import re


class Dataset:
    def __init__(self, data_file, config, train=True):
        with open(data_file, 'r') as fh:
            self.data = json.load(fh)
        self.data_size = len(self.data)
        self.indices = list(range(self.data_size))
        self.train = train
        self.config = config
        if train:
            self.max_len = config.train_max_len
        else:
            self.max_len = config.dev_max_len

    def gen_batches(self, batch_size, shuffle=True, pad_id=0):
        if shuffle:
            np.random.shuffle(self.indices)
        for batch_start in np.arange(0, self.data_size, batch_size):
            batch_indices = self.indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(batch_indices, pad_id)

    def _one_mini_batch(self, indices, pad_id):
        article_word, article_mask, article_len = self.dynamic_padding('article_tokens', indices, pad_id)
        title_word, title_mask, title_len = self.dynamic_padding('title_tokens', indices, pad_id)
        template_word, template_mask, template_len = self.dynamic_padding('template_tokens', indices, pad_id)

        article_ids = [self.data[i]['art_idx'] for i in indices]
        template_ids = [self.data[i]['tp_idx'] for i in indices]

        res = (torch.LongTensor(article_word), torch.LongTensor(title_word), torch.LongTensor(template_word),
               torch.FloatTensor(article_mask), torch.FloatTensor(title_mask), torch.FloatTensor(template_mask),
               article_ids, template_ids)

        return res

    def dynamic_padding(self, key_word, indices, pad_id, max_len=10, ischar=False):
        sample = []
        length = []
        for i in indices:
            sample.append(self.data[i][key_word])
            l = len(self.data[i][key_word])
            max_len = max(max_len, l)
            length.append(l)
        if ischar:
            pads = [pad_id] * self.config.word_len
            pad_sample = [ids + [pads] * (max_len - len(ids)) for ids in sample]
            return pad_sample
        else:
            pad_sample = [ids + [pad_id] * (max_len - len(ids)) for ids in sample]
            mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in sample]
            return pad_sample, mask, length

    def __len__(self):
        return self.data_size


def format_sentences(sentence):
    s = sentence.lower()
    s = re.sub(r"[^0-9a-z]", ' ', s)
    s = re.sub(r"\s+", ' ', s)
    s = s.strip()
    return s


def train(config, device):
    logger = get_logger(config.train_log)
    # read file including: word embedding dict
    word_mat = np.array(np.load(config.word_emb_file), dtype=np.float32)

    logger.info("Building model...")

    train_dataset = Dataset(config.train_token_file, config)
    train_it_num = len(train_dataset) // config.batch_size
    dev_dataset = Dataset(config.dev_token_file, config, train=False)
    dev_it_num = len(dev_dataset) // config.val_batch_size

    model = FastRerank(config.word_dim, word_mat, config.kernel_size, config.encoder_block_num).to(device)
    if config.model:
        model.load_state_dict(torch.load(os.path.join(config.save_dir, config.model)))
    model.train()
    parameters = filter(lambda param: param.requires_grad, model.parameters())

    optimizer = optim.Adam(weight_decay=config.L2_norm, params=parameters, lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)
    loss_func = torch.nn.MarginRankingLoss(config.margin)

    steps = 0
    patience = 0
    losses = 0
    min_loss = 10000
    start_time = time.time()

    for epoch in range(config.epochs):
        batches = train_dataset.gen_batches(config.batch_size, shuffle=True)
        for batch in tqdm(batches, total=train_it_num):
            optimizer.zero_grad()
            (article_word, title_word, template_word, article_mask, title_mask, template_mask, article_ids,
             template_ids) = batch
            article_word, title_word, template_word, article_mask, title_mask, \
            template_mask = article_word.to(device), title_word.to(device), template_word.to(device), \
                            article_mask.to(device), title_mask.to(device), template_mask.to(device)

            gold_scores = model(article_word, title_word, article_mask, title_mask)
            neg_scores = model(article_word, template_word, article_mask, template_mask)
            y = torch.ones_like(gold_scores)
            loss = loss_func(gold_scores, neg_scores, y)

            losses += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
            optimizer.step()
            scheduler.step()

            if (steps + 1) % config.checkpoint == 0:
                losses = losses / config.checkpoint
                logger.info(f"Iteration {steps} train loss {losses}")
                losses = 0
                batches = dev_dataset.gen_batches(config.val_batch_size, shuffle=False)
                for batch in tqdm(batches, total=dev_it_num):
                    (article_word, title_word, template_word, article_mask, title_mask, template_mask, article_ids,
                     template_ids) = batch
                    article_word, title_word, template_word, article_mask, title_mask, \
                    template_mask = article_word.to(device), title_word.to(device), template_word.to(device), \
                                    article_mask.to(device), title_mask.to(device), template_mask.to(device)

                    gold_scores = model(article_word, title_word, article_mask, title_mask)
                    neg_scores = model(article_word, template_word, article_mask, template_mask)
                    y = torch.ones_like(gold_scores)
                    loss = loss_func(gold_scores, neg_scores, y)
                    losses += loss.item()
                losses /= dev_it_num
                logger.info(f"Iteration {steps} dev loss {losses}")

                if losses < min_loss:
                    patience = 0
                    min_loss = losses
                    fn = os.path.join(config.save_dir, f"model_{min_loss}.pkl")
                    torch.save(model.state_dict(), fn)
                else:
                    print(f"patience is {patience}")
                    patience += 1
                    if patience > config.early_stop:
                        logger.info('early stop because val loss is continue increasing!')
                        end_time = time.time()
                        logger.info(f"total training time {end_time - start_time}")
                        exit()
                losses = 0
            steps += 1
    fn = [os.path.join(config.save_dir, "model_final.pkl")]
    torch.save(model.state_dict(), fn)


def dev(config, device):
    keyword = config.keyword
    logger = get_logger(config.dev_log)
    word_mat = np.array(np.load(config.word_emb_file), dtype=np.float32)
    logger.info("Building dev/test model...")
    if keyword == 'train':
        dev_dataset = Dataset(config.train_token_file, config, train=False)
        article = open(config.train_article, 'r').readlines()
        dev_title = open(config.train_title, 'r').readlines()
    elif keyword == 'dev':
        dev_dataset = Dataset(config.dev_token_file, config, train=False)
        article = open(config.dev_article, 'r').readlines()
        dev_title = open(config.dev_title, 'r').readlines()
    else:
        dev_dataset = Dataset(config.test_token_file, config, train=False)
        article = open(config.test_article, 'r').readlines()
        dev_title = open(config.test_title, 'r').readlines()
    dev_it_num = len(dev_dataset) // config.val_batch_size
    batches = dev_dataset.gen_batches(config.val_batch_size, shuffle=False)

    model = FastRerank(config.word_dim, word_mat, config.kernel_size, config.encoder_block_num).to(device)
    if not config.model:
        raise Exception('Empty parameter of --model')
    model.load_state_dict(torch.load(os.path.join(config.save_dir, config.model)))
    model.eval()

    template = []
    rewrite_sample = []
    template_txt = open(config.template_save, 'w')

    temp = open(config.train_title, 'r').readlines()
    for batch in tqdm(batches, total=dev_it_num):
        (article_word, title_word, template_word, article_mask, title_mask, template_mask, article_ids,
         template_ids) = batch
        article_word, title_word, template_word, article_mask, title_mask, \
        template_mask = article_word.to(device), title_word.to(device), template_word.to(device), \
                        article_mask.to(device), title_mask.to(device), template_mask.to(device)
        neg_scores = model(article_word, template_word, article_mask, template_mask)
        neg_scores = neg_scores.view(-1, config.template_num)
        _, index = torch.max(neg_scores, dim=1)
        for i in range(len(index)):
            idx = index[i] + config.template_num*i
            id = template_ids[idx]
            template.append(temp[id])
            template_txt.write(temp[id])
            sample = {'article': article[article_ids[idx]], 'title': dev_title[article_ids[idx]], 'template': temp[id]}
            rewrite_sample.append(sample)

    if keyword == 'train':
        json.dump(rewrite_sample, open(config.train_template, 'w'))
    elif keyword == 'dev':
        json.dump(rewrite_sample, open(config.dev_template, 'w'))
    else:
        json.dump(rewrite_sample, open(config.test_template, 'w'))

    template_txt.close()

    p = open(config.pred_file, 'w')
    new = open(config.template_save, 'r').readlines()
    for n in new:
        p.write(format_sentences(n)+'\n')
