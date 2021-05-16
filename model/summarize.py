import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.corpus import stopwords

from .dataset import get_tokenizer

stopwords = set(stopwords.words('english'))

class Summarizer:
    def __init__(self, task_config, lm='bert', lm_path=None):
        self.config = task_config
        self.tokenizer = get_tokenizer(lm, lm_path)
        self.len_cache = {}

    def build_index(self, contents):
        content = []
        for line in contents:
            LL = line.split('\t')
            if len(LL) > 2:
                tokens = LL[1].split(' ')
                filter_tokens = []
                for token in tokens:
                    if token not in ['COL', 'VAL'] and \
                       token not in stopwords:
                        filter_tokens.append(token)
                content.append(' '.join(filter_tokens))

        vectorizer = TfidfVectorizer().fit(content)
        self.vocab = vectorizer.vocabulary_
        self.idf = vectorizer.idf_

    def get_len(self, word):
        if word in self.len_cache:
            return self.len_cache[word]
        length = len(self.tokenizer.tokenize(word))
        self.len_cache[word] = length
        return length

    def transform(self, row, max_len=128):
        sentA, sentB, label = row.strip().split('\t')
        res = sentA + '\t'

        for sent in [sentB]:
            attr_sent = ' '.join(
                filter(lambda x: len(x),
                       map(lambda x: re.sub('COL .*', '', x).strip(),
                           sent.split('VAL'))))

            token_cnt = Counter(attr_sent.split(' '))
            total_len = 0

            subset = Counter()

            for token, cnt in token_cnt.most_common():
                if token in self.vocab:
                    # attribute name
                    if cnt == 1 and self.idf[self.vocab[token]] == 1 :
                        continue

                    subset[token] = cnt / self.idf[self.vocab[token]]
            subset = subset.most_common(max_len)

            # Remove own token
            i = 0
            prev = 0
            for _, cnt in subset:
                if cnt != 1.0 and prev == cnt:
                    break
                prev = cnt
                i += 1

            if i != len(subset):
                subset = subset[:i-1]

            topk_tokens_copy = set([])
            for word, _ in subset:
                bert_len = self.get_len(word)
                if total_len + bert_len > max_len:
                    break
                total_len += bert_len
                topk_tokens_copy.add(word)

            name_flag = 0
            for token in sent.split(' '):
                if token in ['COL', 'VAL']:
                    res += token + ' '
                    if token == 'COL':
                        name_flag = 1
                elif name_flag:
                    res += token + ' '
                    name_flag = 0
                elif token in topk_tokens_copy:
                    res += token + ' '
                    topk_tokens_copy.remove(token)

            res += '\t'

        res += label + '\n'
        return res

    def transform_file(self, input_fn, batch_size=17, max_len=256, overwrite=False):
        out_fn = input_fn + '.su'
        if not os.path.exists(out_fn) or \
           os.stat(out_fn).st_size == 0 or overwrite:
            with open(out_fn, 'w') as fout:
                batch = []

                for line in open(input_fn):
                    batch.append(line)

                    if len(batch) == batch_size:
                        self.build_index(batch)
                        for line in batch:
                            fout.write(self.transform(line, max_len=max_len))
                        batch.clear()
        return out_fn
