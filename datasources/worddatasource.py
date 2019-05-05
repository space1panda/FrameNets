import numpy as np
import re
from itertools import count
from utils.utils_np import chunks


class WordDatasource:

    def __init__(self, path, seq_len, max_count):

        """ Initialize vocabularies for word rnn lstm """

        self._token2ix = {None:0,'\n': 1}
        self._ix2token = [None,'\n']
        self._vocab_size = len(self._token2ix)
        self._stats = {}
        self._sequences = []
        self._targets = []

        data = open(path, 'r').read().lower()
        data = re.sub('[^a-zA-Z\n ]+', '', data)
        data = data.replace('\n', ' ').split(' ')
        data = [d for d in data if d]

        for ix, i in enumerate(data):
            if i not in self._stats:
                self._stats.update({i: 0})
            if self._stats[i] <= max_count:
                self._stats[i] += 1
            else:
                del data[ix]

        source = chunks(data, seq_len)

        next_ix = count(2, 1)
        while True:
            try:
                s = next(source)
                for t in s:
                    if t not in self._token2ix:
                        ix = next(next_ix)
                        self._token2ix.update({t: ix})
                        self._ix2token.append(t)
                if s not in self._sequences:
                    self._sequences.append([self._token2ix[None]]+[self._token2ix[t] for t in s])
                    self._targets.append([self._token2ix[t] for t in s]+[self._token2ix['\n']])
            except StopIteration:
                break


        self._sequences = np.array(self._sequences[:-1])
        self._targets = np.array(self._targets[:-1])

    def __getitem__(self, idx):
        return self._sequences[idx], self._targets[idx]

    def __len__(self):
        return len(self._sequences)