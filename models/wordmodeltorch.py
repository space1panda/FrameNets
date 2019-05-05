import torch
import torch.nn as nn
import torch.nn.functional as F


class WordModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, bi=False, inference=False):

        super(WordModel, self).__init__()
        self.inference = inference
        self._num_classes = vocab_size
        self._hidden_dim_out = hidden_dim * max(1, int(bi)+1)
        self._word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self._lstm = nn.LSTM(input_size=embedding_dim,
                             hidden_size=hidden_dim,
                             num_layers=1,
                             batch_first=True,
                             bidirectional=bi)
        
        # FIXME: bidirectional not working?
        self._fc1 = nn.Linear(self._hidden_dim_out, vocab_size)
        if self.inference:
            self._h_n, self._c_n = torch.zeros((1,1, hidden_dim)), torch.zeros((1,1, hidden_dim))

    def forward(self, x):
        x_embeds = self._word_embeddings(x)
        if self.inference:
            lstm_out, (self._h_n, self._c_n) = self._lstm(x_embeds, (self._h_n, self._c_n))
        else:
            lstm_out, _ = self._lstm(x_embeds)
        tag_space = self._fc1(lstm_out.contiguous().view(-1, self._hidden_dim_out))
        tag_scores = F.softmax(tag_space, dim=1)
        tag_scores = tag_scores.view(-1, self._num_classes)
        return tag_scores