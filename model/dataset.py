import random
import re

import numpy as np
import torch

from torch.utils import data

# from .argument import Augmenter

tokenizer = None


def get_lm_path(lm, lm_path):
    if lm_path != None:
        return lm_path

    if lm == 'bert':
        return 'bert-base-uncased'
    elif lm == 'distilbert':
        return 'distilbert-base-uncased'
    elif lm == 'roberta':
        return 'roberta-base'
    elif lm == 'xlnet':
        return 'xlnet-base-cased'


def get_tokenizer(lm, lm_path):
    global tokenizer

    path = get_lm_path(lm, lm_path)
    if tokenizer is None:
        if lm == 'bert':
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained(path)
        elif lm == 'distilbert':
            from transformers import DistilBertTokenizer
            tokenizer = DistilBertTokenizer.from_pretrained(path)
        elif lm == 'roberta':
            from transformers import RobertaTokenizer
            tokenizer = RobertaTokenizer.from_pretrained(path)
        elif lm == 'xlnet':
            from transformers import XLNetTokenizer
            tokenizer = XLNetTokenizer.from_pretrained(path)

    return tokenizer


class Dataset(data.Dataset):
    def __init__(self, source, category, lm='bert', lm_path=None, max_len=512, split=True, augment_op=None):
        self.tokenizer = get_tokenizer(lm, lm_path)

        # tokens and tags
        self.max_len = max_len
        self.augment_op = augment_op # Augmentation is not used in the currently.
        # if augment_op != None:
        #     self.augmenter = Augmenter()
        # else:
        self.augmenter = None

        sents, tags_li, attributes = self.read_classification_file(source, split)

        # assign class variables
        self.sents, self.tags_li, self.attributes = sents, tags_li, attributes
        self.category = category

        self.attr_num = len(self.attributes[0][0])

        # index for tags/labels
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.category)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.category)}

    def read_classification_file(self, path, split):
        sents, labels, attributes = [], [], []
        for line in open(path):
            items = line.strip().split('\t')

            if self.augmenter != None:
                aug_items = [self.augmenter.augment_sent(item, self.augment_op)
                             for item in items[0:-1]]
                items[0:-1] = aug_items

            attrs = []
            if split:
                attr_items = [item + ' COL' for item in items[0:-1]]
                for attr_item in attr_items:
                    attrs.append([f"COL {attr_str}" for attr_str
                                  in re.findall(r"(?<=COL ).*?(?= COL)", attr_item)])
                assert len(attrs[0]) == len(attrs[1])
            else:
                attrs = [[item] for item in items[0:-1]]

            sents.append(items[0] + ' [SEP] ' + items[1])
            labels.append(items[2])
            attributes.append(attrs)
        return sents, labels, attributes

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags, attributes = self.sents[idx], self.tags_li[idx], self.attributes[idx]

        xs = [self.tokenizer.encode(text=attributes[0][i], text_pair=attributes[1][i],
                                    add_special_tokens=True, truncation="longest_first", max_length=self.max_len)
              for i in range(self.attr_num)]
        left_zs = [self.tokenizer.encode(text=attributes[0][i], add_special_tokens=True,
                                         truncation="longest_first", max_length=self.max_len)
                   for i in range(self.attr_num)]
        right_zs = [self.tokenizer.encode(text=attributes[1][i], add_special_tokens=True,
                                          truncation="longest_first", max_length=self.max_len)
                    for i in range(self.attr_num)]

        masks = [torch.zeros(self.tokenizer.vocab_size, dtype=torch.int)
                 for _ in range(self.attr_num)]
        for i in range(self.attr_num):
            masks[i][xs[i]] = 1
        masks = torch.stack(masks)

        y = self.tag2idx[tags]  # label

        seqlens = [len(x) for x in xs]
        left_zslens = [len(left_z) for left_z in left_zs]
        right_zslens = [len(right_z) for right_z in right_zs]

        return words, xs, y, seqlens, masks, left_zs, right_zs, left_zslens, right_zslens, attributes

    def get_attr_num(self):
        return self.attr_num

    @staticmethod
    def pad(batch):
        f = lambda x: [sample[x] for sample in batch]
        g = lambda x, seqlen, val: \
            [[sample + [val] * (seqlen - len(sample)) \
              for sample in samples[x]]
             for samples in batch]  # 0: <pad>

        # get maximal sequence length
        seqlens = f(3)
        maxlen = np.array(seqlens).max()

        words = f(0)
        xs = torch.LongTensor(g(1, maxlen, 0))
        y = f(2)
        masks = torch.stack(f(4))

        if isinstance(y[0], float):
            y = torch.Tensor(y)
        else:
            y = torch.LongTensor(y)
        return words, xs, y, seqlens, masks

    @staticmethod
    def padJoin(batch):
        f = lambda x: [sample[x] for sample in batch]
        g = lambda x, seqlen, val: \
            [[sample + [val] * (seqlen - len(sample)) \
              for sample in samples[x]]
             for samples in batch]  # 0: <pad>

        # get maximal sequence length
        seqlens = f(3)
        maxlen = np.array(seqlens).max()

        words = f(0)
        xs = torch.LongTensor(g(1, maxlen, 0))
        y = f(2)
        masks = torch.stack(f(4))

        attributes = f(9)
        attr_num = xs.size()[1]

        right_attributes = []
        for i in range(attr_num):
            right_attribute = []
            for attribute in attributes:
                right_attribute.append(attribute[1][i])
            right_attributes.append(right_attribute)

        zs = [tokenizer.encode(text=' '.join(right_attributes[i]),
                    add_special_tokens=False, truncation="longest_first", max_length=512)
              for i in range(attr_num)]
        maxlen = np.array([len(z) for z in zs]).max()
        zs = [z + [0] * (maxlen-len(z)) for z in zs]
        zs = torch.LongTensor(zs).unsqueeze(0).permute(1, 0, 2)


        if isinstance(y[0], float):
            y = torch.Tensor(y)
        else:
            y = torch.LongTensor(y)

        return words, xs, zs, y, seqlens, masks
