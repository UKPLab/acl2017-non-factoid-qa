import math

import numpy as np

from experiment.qa.train import QABatchedTraining


class QATrainingSimple(QABatchedTraining):
    """This is a simple training method that runs over the training data in a linear fashion, just like in keras."""

    def __init__(self, config, config_global, logger):
        super(QATrainingSimple, self).__init__(config, config_global, logger)
        self._train_questions, self._train_answers_good, self._train_answers_bad = [], [], []
        self.batch_i = 0
        self.epoch_random_indices = []

    def prepare_next_epoch(self, model, data, sess, epoch):
        """Prepares the next epoch, especially the batches"""
        super(QATrainingSimple, self).prepare_next_epoch(model, data, sess, epoch)
        self.batch_i = 0

        # training examples are prepared in the first epoch
        if len(self._train_questions) == 0:
            self.logger.debug('Preparing training examples')
            self._train_questions, self._train_answers_good, self._train_answers_bad = data.get_items(
                data.archive.train.qa,
                self.negative_answers
            )

        # shuffle the indices of each batch
        self.epoch_random_indices = np.random.permutation(len(self._train_questions))

    def get_n_batches(self):
        return math.ceil(len(self._train_questions) / self.batchsize)

    def get_next_batch(self, model, data, sess):
        """Return the training data for the next batch

        :return: questions, good answers, bad answers
        :rtype: list, list, list
        """
        indices = self.epoch_random_indices[self.batch_i * self.batchsize: (self.batch_i + 1) * self.batchsize]

        batch_questions = [self._train_questions[i] for i in indices]
        batch_answers_good = [self._train_answers_good[i] for i in indices]
        batch_answers_bad = [self._train_answers_bad[i] for i in indices]
        self.batch_i += 1

        return batch_questions, batch_answers_good, batch_answers_bad


component = QATrainingSimple
