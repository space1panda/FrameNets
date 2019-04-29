from utils.charutils_np import *
from datasources.chardataloader import *
from models.charmodel_np import *
import pickle


class CharTrainer:

    def __init__(self, lr, n_epochs, hidden, vocab, batch_size, seq_len,
                 clip_ratio, path, save_path, **kwargs):
        self._lr = lr
        self._n_epochs = n_epochs
        self._hidden = hidden
        self._vocab = vocab
        self._batch_size = batch_size
        self.save_path = save_path

        self._parameters = initialize_parameters(self._hidden, self._vocab)

        self._dataload = RNNDataLoader(batch_size=batch_size,
                                       path=path, seq_len=seq_len)

        self._model = CharModelRNN(clip_ratio, self._parameters, seq_len)

    def train(self):
        train_epoch_loss = {}

        batches = self._dataload._batches
        loss = get_initial_loss(self._vocab, self._batch_size)
        hidden_back = np.zeros((self._hidden, self._batch_size))

        for j in range(self._n_epochs):

            for index in range(len(batches)):
                X = batches[index][0]
                Y = batches[index][1]

                curr_loss, gradients, hidden_back = self._model.optimize(X, Y,
                                                                         hidden_back)

                curr_loss = np.mean(curr_loss)
                loss = smooth(loss, curr_loss)
                train_epoch_loss.update({j: loss})

                self._parameters = update_parameters_lronly(self._parameters,
                                                            gradients, self._lr)

            if j % (self._n_epochs / 10) == 0:
                print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

        # TODO refactor saving method
        f = open(self.save_path, "wb")
        pickle.dump(self._parameters, f)
        f.close()
        return self._parameters


