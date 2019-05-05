from datasources.worddatasource import WordDatasource
from models.wordmodeltorch import WordModel
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.optim as optim
import numpy as np


class WordTrainer:

    def __init__(self, lr, n_epochs, embedding_dim, hidden_dim,
                 batch_size, seq_len, path, save_path, criterion,
                 split_factor, max_count, bi=False, **kwargs):

        self._n_epochs = n_epochs
        self._hidden = hidden_dim
        self._batch_size = batch_size
        self.save_path = save_path
        self._criterion = criterion


        datasource = WordDatasource(path, seq_len, max_count)
        length = datasource.__len__()
        idxs = np.arange(length)

        train_set = Subset(datasource, idxs[:int(length * split_factor)])
        test_set = Subset(datasource, idxs[int(length * split_factor):])

        self._train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                                  shuffle=True)

        self._test_loader = DataLoader(dataset=test_set, batch_size=batch_size,
                                 shuffle=True)

        self._model = WordModel(embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                          vocab_size=len(datasource._token2ix), bi=False)

        self._optimizer = optim.Adam(self._model.parameters(), lr=lr)

    def _train(self):

        losses = {}

        for e in range(self._n_epochs):

            self._model.train()
            train_loss = []
            with tqdm(total=len(self._train_loader)) as bar:
                for idx, (inputs, labels) in enumerate(self._train_loader):
                    self._optimizer.zero_grad()
                    output = self._model(inputs)
                    loss = self._criterion(output, labels.view(-1))
                    loss.backward()
                    self._optimizer.step()
                    train_loss.append(loss.data.item())
                    bar.set_description('[train epoch {}]'.format(e))
                    bar.set_postfix(str="train loss %.3f" % np.mean(train_loss))
                    bar.update()


            self._model.eval()
            valid_loss = []
            with tqdm(total=len(self._test_loader)) as bar:
                for idx, (inputs, labels) in enumerate(self._test_loader):

                    with torch.no_grad():
                        output = self._model(inputs)

                    loss = self._criterion(output, labels.view(-1))
                    valid_loss.append(loss.data.item())
                    bar.set_description('[valid epoch {}]'.format(e))
                    bar.set_postfix(str="valid loss %.3f" % np.mean(valid_loss))
                    bar.update()
            losses.update({e: np.mean(valid_loss)})

        torch.save({'model_state':self._model.state_dict()}, self.save_path)


