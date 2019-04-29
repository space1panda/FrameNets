from datasources.chardatasource import CharDatasource
from utils.charutils_np import chunks

class RNNDataLoader(CharDatasource):

    def __init__(self, batch_size, **kwargs):

        """ CharDatasource is used as parent object.
        train/test mode defines which part of dataset to use for batching
        """

        self._batch_size = batch_size
        super().__init__(**kwargs)

        """Just to indicate - we are encountering chunks utility function for the second time.
        In project this should go to utility - otherwise it's a bad practice
        """

        """Splitting dataset into train/test
        """

        train_set = (self._tokens, self._targets)

        """Even though we are having pretty small dataset, it's a 
        good practice to use generators instead of keeping
        additional data arrays to free up memory
        """

        x_loader = chunks(train_set[0], self._batch_size)
        y_loader = chunks(train_set[1], self._batch_size)

        self._batches = []

        while True:
            try:
                x, y = next(x_loader), next(y_loader)
                if len(x) == self._batch_size:
                    self._batches.append((x, y))

            except StopIteration:
                break

    def _getbatch(self, idx):
        return self._batches[idx]