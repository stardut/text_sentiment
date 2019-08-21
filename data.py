# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd


class dataset(object):
    """docstring for data"""

    def __init__(self, datafile, max_len=10):
        super(dataset, self).__init__()
        self.char_map_file = 'data/char_map.json'
        self.train_file = 'data/train.csv'
        self.test_file = 'data/test.csv'
        self.val_file = 'data/val.csv'
        self.datafile = datafile
        self.max_len = max_len
        self.vocab = {}
        self.char_num = 1
        self.pointer = 0
        self.load_data()

    def addChar(self, char):
        if char not in self.vocab:
            self.vocab[char] = self.char_num
            self.char_num += 1

    def addSeq(self, seq):
        for char in seq:
            self.addChar(char)

    def split_data(self, datafile, rate=(0.7, 0.15, 0.15)):
        print('split train, test, val data...')
        df = pd.read_csv(datafile)
        labels = set(df.label)
        train, test, val = [], [], []

        for label in labels:
            ls = df[df.label == label]
            idx1 = int(len(ls) * rate[0])
            idx2 = int(len(ls) * (rate[0] + rate[1]))
            train.append(ls[:idx1])
            test.append(ls[idx1:idx2])
            val.append(ls[idx2:])

        if os.path.exists('data/'):
            os.mkdir('data/')
        self.save_split_data(train, self.train_file)
        self.save_split_data(test, self.test_file)
        self.save_split_data(val, self.val_file)

    def save_split_data(slef, data, path):
        df = pd.concat(data)
        df.to_csv(path, index=0)

    def gen_char_map(self):
        df = pd.read_csv(self.datafile, encoding='utf-8-sig')
        for line in df['review'].values:
            self.addSeq(line)

        with open(self.char_map_file, 'w+', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False)

    def train_data(self):
        yield [inputs, targets]

    def char2id(self, char):
        return self.vocab[char]

    def load_char_map(self):
        with open(self.char_map_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
            self.vocab_len = len(self.vocab)

    def load_data(self):
        def shuffle(df):
            return df.sample(frac=1).reset_index(drop=True)

        df = shuffle(pd.read_csv(self.datafile, encoding='utf-8-sig'))
        self.load_char_map()
        self.df2rundata(df)

    def seq2ids(self, seq):
        pading_list = [0] * self.max_len
        ids = list(map(self.char2id, seq))
        ids += pading_list
        return np.array(ids[:self.max_len])

    def df2rundata(self, df):
        lines = df['review'].to_numpy()
        self.inputs = np.array(list(map(self.seq2ids, lines)))
        self.labels = df['label'].to_numpy()

    def next(self, batch_size=64):
        while 1:
            s = self.pointer
            e = s + batch_size
            if e > len(self.labels):
                self.pointer = 0
                break

            inputs = self.inputs[s:e]
            labels = self.labels[s:e]
            yield inputs, labels
            self.pointer = e


data = dataset('data/simplifyweibo_4_moods.csv')
data.gen_char_map()
