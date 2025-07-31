from collections import Counter

import numpy as np
import torch
from nltk import word_tokenize
from torch.autograd import Variable

# Import global constants from the new config file
from src.config import BATCH_SIZE, UNK, PAD, DEVICE


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = torch.from_numpy(src).to(DEVICE).long()
        self.src_mask = (self.src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = torch.from_numpy(trg).to(DEVICE).long()[:, :-1]
            self.trg_y = torch.from_numpy(trg).to(DEVICE).long()[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(trg, pad):
        trg_mask = (trg != pad).unsqueeze(-2)
        trg_mask = trg_mask & Variable(subsequent_mask(trg.size(-1)).type_as(trg_mask.data))
        return trg_mask


class PrepareData:
    def __init__(self, train_file, dev_file):
        self.train_en_raw, self.train_cn_raw = self.load_data(train_file)
        self.dev_en_raw, self.dev_cn_raw = self.load_data(dev_file)

        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(
            self.train_en_raw + self.dev_en_raw)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(
            self.train_cn_raw + self.dev_cn_raw)

        self.train_en, self.train_cn = self.wordToID(
            self.train_en_raw, self.train_cn_raw, self.en_word_dict, self.cn_word_dict)
        self.dev_en, self.dev_cn = self.wordToID(
            self.dev_en_raw, self.dev_cn_raw, self.en_word_dict, self.cn_word_dict)

        self.train_data = self.splitBatch(self.train_en, self.train_cn, BATCH_SIZE)
        self.dev_data = self.splitBatch(self.dev_en, self.dev_cn, BATCH_SIZE)

    def load_data(self, path):
        en = []
        cn = []
        with open(path, 'r', encoding='utf-8') as fin:
            for line in fin:
                list_content = line.strip().split('\t')
                if len(list_content) < 2:
                    continue
                en.append(['BOS'] + word_tokenize(list_content[0]) + ['EOS'])
                cn.append(['BOS'] + word_tokenize(list_content[1]) + ['EOS'])
        return en, cn

    def build_dict(self, sentences, max_words=50000):
        word_count = Counter()
        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1

        ls = word_count.most_common(max_words)
        total_words = len(ls) + 2

        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = UNK
        word_dict['PAD'] = PAD

        index_dict = {v: k for k, v in word_dict.items()}
        return word_dict, total_words, index_dict

    def wordToID(self, en, cn, en_dict, cn_dict, sort=True):
        out_en_ids = [[en_dict.get(w, UNK) for w in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(w, UNK) for w in sent] for sent in cn]

        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        if sort:
            sorted_index = len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[i] for i in sorted_index]
            out_cn_ids = [out_cn_ids[i] for i in sorted_index]
        return out_en_ids, out_cn_ids

    def splitBatch(self, en, cn, batch_size, shuffle=True):
        idx_list = np.arange(0, len(en), batch_size)
        if shuffle:
            np.random.shuffle(idx_list)

        batches = []
        for idx in idx_list:
            batch_index = np.arange(idx, min(idx + batch_size, len(en)))
            batch_en = [en[index] for index in batch_index]
            batch_cn = [cn[index] for index in batch_index]

            batch_cn = seq_padding(batch_cn)
            batch_en = seq_padding(batch_en)
            batches.append(Batch(batch_en, batch_cn))
        return batches
