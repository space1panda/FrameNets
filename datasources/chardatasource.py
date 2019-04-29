import numpy as np
import re

class CharDatasource:

    def __init__(self, path, seq_len, *args):

        """Initialize vocabularies for char rnn (w||w/o lstm).
        In our case we want to train only characters and < (end of word) token
         Cleaning data with regex: \n character left to indicate end of token(word)
        to train  ending word, cleaning all symbols and spaces
        """

        data = open(path, 'r').read().lower()
        data = re.sub('[^0-9a-zA-Z\n]+', '', data)
        chars = list(set(data))

        self._seq_len = seq_len
        self._char2ix = {ch: i for i, ch in enumerate(sorted(chars))}
        self._ix2char = {i: ch for i, ch in enumerate(sorted(chars))}
        self._vocab_size = len(self._char2ix)

        """Initialize vocabulary with char appearance statistics. 
        It can be useful to determine unfrequently used characters to omit them in training. 
        In this experiment however we are not excluding characters due to small dataset.  
        """

        self._char_stats = {char: 0 for char in sorted(chars)}

        # Initialize x, y sets as empty lists

        self._tokens = []
        self._targets = []

        """generator function for getting dataset  desired sequence length. We are not setting 
        sequence length to the length of token to apply vectorization without padding.
        This is a utility function which is not supposed to be inside of the object
        """

        def chunks(l, n):
            for i in range(0, len(l), n): yield l[i:i + n]

        """Call generator to encode input and output sequences. 
        Output is a one-step forward sequence in the same datasource - 
        We want our model to learn predicting next character
        """

        source = chunks(data, seq_len)

        while True:
            try:
                s = next(source)
                for ch in s:
                    self._char_stats[ch] += 1
                self._tokens.append([self._char2ix['\n']] + [self._char2ix[ch] for ch in s])
                self._targets.append([self._char2ix[ch] for ch in s] + [self._char2ix['\n']])

            except StopIteration:
                break

        """Vectorize with numpy and quickly one-hot the input set.
        Final shapes = tokens - (num_of_sequences, sequence length, vocab_size)
        targets - (num_of_sequences, sequence length, char2idx[character])
        We are not creating one-hot for outputs because we want to count loss 
        between only most-probable character index and target
        """

        self._tokens = np.array(self._tokens[:-1], dtype=np.int32)
        self._targets = np.array(self._targets[:-1])
        x_values = np.max(self._tokens) + 1
        self._tokens = np.eye(x_values)[self._tokens]

    # TODO @properties

        """the following methods are useful for possible integrations with PyTorch objects"""

    def __getitem__(self, idx):
        return self._tokens[idx], self._targets[idx]

    def __len__(self):
        return len(self._tokens)
