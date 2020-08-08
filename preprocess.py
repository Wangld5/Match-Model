from codecs import open
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np

nlp = spacy.blank("en")


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def process_file(config, source, target, index_file, data_type, word_counter, title):
    print(f"Generating {data_type} example...")
    examples = []

    def filter_func(example):
        if data_type == 'train':
            return len(example) > config.train_max_len
        else:
            return False

    # read file:(article, title, template)
    with open(source, 'r') as fh:
        articles = fh.readlines()
    with open(target, 'r') as fh:
        templates = fh.readlines()
    with open(title, 'r') as fh:
        titles = fh.readlines()
    with open(index_file, 'r') as fh:
        dataset_index = json.load(fh)
    f = 0
    for indexs in tqdm(dataset_index):

        art_idx = int(indexs['art_idx'])
        article = articles[art_idx]
        article = article.replace("''", '" ').replace("``", '" ')
        article_tokens = word_tokenize(article)
        for token in article_tokens:
            word_counter[token] += 1
        title = titles[art_idx]
        title = title.replace("''", '" ').replace("``", '" ')
        title_tokens = word_tokenize(title)
        for index in indexs['tp_idx']:
            template = templates[int(index)]
            template = template.replace("''", '" ').replace("``", '" ')
            tp_tokens = word_tokenize(template)
            for token in tp_tokens:
                word_counter[token] += 1
            example = {"article_tokens": article_tokens,
                       "title_tokens": title_tokens,
                       "template_tokens": tp_tokens,
                       "tp_idx": int(index),
                       "art_idx": art_idx}
            examples.append(example)
    print(f"{len(examples)} samples in total")
    return examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, vec_size=None):
    print(f"{data_type} generating embedding...")
    embedding_dict = {}
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, 'r', encoding="utf-8") as fh:
            for line in tqdm(fh):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print(f"{len(embedding_dict)}/{len(counter)} tokens have corresponding {data_type} embedding vector")

    for word in counter.keys():
        if word not in embedding_dict:
            embedding_dict[word] = np.random.random(size=vec_size)
    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    idx2token = {idx: token for token, idx in enumerate(token2idx_dict.keys(), 2)}
    return emb_mat, token2idx_dict, idx2token


def build_feature(config, examples, data_type, out_file, word2idx_dict, is_test=False):
    max_len = config.dev_max_len if is_test else config.train_max_len
    template_limit = config.template_max_len
    title_limit = config.title_max_len

    def filter_func(example):
        if is_test:
            return False
        return len(example["article_tokens"]) > max_len or len(example["template_tokens"]) > template_limit

    print(f"processing {data_type} examples...")
    total = 0
    meta = {}
    N = len(examples)
    example_ids = []
    for n, example in tqdm(enumerate(examples)):
        new_example = {'article_tokens': [], 'template_tokens': [], 'title_tokens': []}
        total += 1

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        for token in example["article_tokens"]:
            new_example["article_tokens"].append(_get_word(token))
        for token in example["template_tokens"]:
            new_example["template_tokens"].append(_get_word(token))
        for token in example["title_tokens"]:
            new_example["title_tokens"].append(_get_word(token))

        if len(new_example["article_tokens"]) > max_len:
            new_example["article_tokens"] = new_example["article_tokens"][:max_len]
        if len(new_example["template_tokens"]) > template_limit:
            new_example["template_tokens"] = new_example["template_tokens"][:template_limit]
        if len(new_example["title_tokens"]) > title_limit:
            new_example["title_tokens"] = new_example["title_tokens"][:title_limit]
        new_example['art_idx'] = example['art_idx']
        new_example['tp_idx'] = example['tp_idx']
        example_ids.append(new_example)
    save(out_file, example_ids, message=out_file)
    print(f"build {total} / {N} instances of features in total")
    meta["total"] = total
    return meta


def save(out_file, obj, message=None):
    if message is not None:
        print(f"saving {message}...")
        with open(out_file, 'w') as f:
            json.dump(obj, f)
        # np.save(out_file, obj)
        print('finish saved')


def save_emb(out_file, obj, message=None):
    if message is not None:
        print(f"saving {message}...")
        np.save(out_file, obj)
        print('finish saved')


def preproc(config):
    word_counter, char_counter = Counter(), Counter()
    # train_examples = process_file(config,
    #                               config.train_article, config.train_title, config.train_dataset_index, 'train',
    #                               word_counter, config.train_title)
    # dev_examples = process_file(config,
    #                             config.dev_article, config.train_title, config.dev_dataset_index, 'dev', word_counter,
    #                             config.dev_title)
    test_examples = process_file(config,
                                 config.test_article, config.train_title, config.test_dataset_index, 'test',
                                 word_counter,
                                 config.test_title)

    word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
    word_emb_mat, word2idx_dict, idx2token = get_embedding(word_counter, 'word', emb_file=None, vec_size=300)
    print(len(word_emb_mat))
    # save_emb(config.word_emb_file, word_emb_mat, message="word embedding")
    build_feature(config, dev_examples, 'dev', config.dev_token_file, word2idx_dict, is_test=True)
    build_feature(config, test_examples, 'test', config.test_token_file, word2idx_dict, is_test=True)
    build_feature(config, train_examples, 'train', config.train_token_file, word2idx_dict)
