from __future__ import division

import math
import os
import shutil

import numpy as np
import progressbar
import tensorflow as tf

import experiment


class QATraining(experiment.Training):
    def __init__(self, config, config_global, logger):
        super(QATraining, self).__init__(config, config_global, logger)

        self.length_question = self.config_global['question_length']
        self.length_answer = self.config_global['answer_length']
        self.n_epochs = self.config['epochs']
        self.n_epochs_per_run = self.config.get('epochs_per_run', self.n_epochs)
        self.batchsize = self.config['batchsize']
        self.batchsize_valid = self.config.get('batchsize_valid', self.batchsize)
        self.dropout_keep_prob = 1.0 - self.config.get('dropout', 0.0)
        self.scorer = self.config.get('scorer', None)
        self.scorer_print = self.config.get('scorer_print', list())

        self.negative_answers = self.config.get('negative_answers', 1)
        self.negative_answers_valid = self.config.get('negative_answers_valid', 50)

        # cached vectors for validation ranking
        self.rank_data = None

        # tensorboard summary writer
        self.global_step = 0
        self.train_writer = None

        # checkpointing and weight restoring
        self.run_recorded_epochs = 0  # number of recorded epochs in this run
        self.state = QATrainState(
            self.config.get('save_folder', None), less_is_better=self.scorer is None, logger=self.logger
        )
        self.early_stopping_patience = self.config.get('early_stopping_patience', self.n_epochs)

    def remove_checkpoints(self):
        self.state.clear()

    def start(self, model, data, sess):
        if 'tensorflow_log_dir' in self.config_global:
            self.train_writer = tf.summary.FileWriter(self.config_global['tensorflow_log_dir'], sess.graph)
        return 0, 0

    def add_summary(self, summary):
        if self.train_writer:
            self.train_writer.add_summary(summary, self.global_step)

    def record_epoch(self, sess, score):
        self.run_recorded_epochs += 1
        previous_score = self.state.best_score
        self.state.record(sess, score)

        if previous_score != self.state.best_score:
            self.logger.info('Validation score improved from {:.6f} to {:.6f}'.format(
                previous_score, self.state.best_score
            ))
        else:
            self.logger.info('Validation score did not improve ({:.6f}; best={:.6f})'.format(
                score, self.state.best_score
            ))

    def restore_best_weights(self, sess):
        self.state.load(sess, weights='best')

    def is_early_stopping(self):
        return self.state.recorded_epochs - self.state.best_epoch > self.early_stopping_patience

    def calculate_validation_scores(self, sess, model, data, scorers):
        """

        :param sess:
        :param model:
        :param data:
        :return: :return: the first return value will be the target-score, all subsequent values are for printing
        """
        if self.scorer is None:
            self.logger.error('Need a configured scorer to calculate validation score')
            raise Exception('Need a configured scorer to calculate validation score')

        ranks, average_precisions = self._calculate_validation_ranks(sess, model, data)
        ranks = [r for r in ranks if r > 0]
        scores = {
            'accuracy': len([r for r in ranks if r == 1]) / float(len(ranks)),
            'mrr': np.mean([1 / float(r) for r in ranks]),
            'map': np.mean(average_precisions),
            'loss': self.calculate_valid_loss(sess, model, data) if 'loss' in scorers else 0.0
        }

        return [scores[s] for s in scorers]

    def _calculate_validation_ranks(self, sess, model, data):
        if self.rank_data is None:
            self.logger.debug('Creating validation ranking data')
            self.rank_data = [(
                pool,
                np.array(
                    [data.get_item_vector(pool.question, self.length_question)] * len(
                        pool.pooled_answers)
                ),
                np.array(
                    [data.get_item_vector(a, self.length_answer) for a in pool.pooled_answers]
                )
            ) for pool in data.archive.valid.qa]

        ranks = []
        average_precisions = []
        bar = _create_progress_bar()
        for pool, valid_questions, valid_answers in bar(self.rank_data):
            scores = []
            for valid_batch in range(int(math.ceil(len(valid_answers) / float(self.batchsize_valid)))):
                valid_batch_indices = self.batchsize_valid * valid_batch, self.batchsize_valid * (valid_batch + 1)
                valid_batch_scores, = sess.run([model.predict], feed_dict={
                    model.input_question: valid_questions[valid_batch_indices[0]:valid_batch_indices[1]],
                    model.input_answer_good: valid_answers[valid_batch_indices[0]:valid_batch_indices[1]],
                    model.dropout_keep_prob: 1.0,
                })
                scores += valid_batch_scores.tolist()

            rank = 0
            precisions = []
            sorted_answers = sorted(zip(scores, pool.pooled_answers), key=lambda x: -x[0])
            for i, (score, answer) in enumerate(sorted_answers, start=1):
                if answer in pool.ground_truth:
                    if rank == 0:
                        rank = i
                    precisions.append((len(precisions) + 1) / float(i))
            ranks.append(rank)
            average_precisions.append(np.mean(precisions))

        return ranks, average_precisions

    def calculate_valid_loss(self, sess, model, data):
        raise NotImplementedError()


class QABatchedTraining(QATraining):
    """This is a simple training method that runs over the training data in a linear fashion, just like in keras."""

    def __init__(self, config, config_global, logger):
        super(QABatchedTraining, self).__init__(config, config_global, logger)
        self.initial_learning_rate = self.config.get('initial_learning_rate', 1.1)
        self.dynamic_learning_rate = self.config.get('dynamic_learning_rate', True)

        self._valid_questions, self._valid_answers_good, self._valid_answers_bad = None, None, None

        self.epoch_learning_rate = self.initial_learning_rate

    def start(self, model, data, sess):
        super(QABatchedTraining, self).start(model, data, sess)

        learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer_name = self.config.get('optimizer', 'sgd')
        if optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif optimizer_name == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9)
        else:
            raise Exception('No such optimizer: {}'.format(optimizer_name))

        train = optimizer.minimize(model.loss)

        self.logger.info('Initializing all variables')
        sess.run(tf.global_variables_initializer())

        self.state.load(sess, weights='last')
        start_epoch = self.state.recorded_epochs + 1
        end_epoch = min(self.n_epochs + 1, start_epoch + self.n_epochs_per_run)

        if self.state.recorded_epochs > 0:
            self.logger.info('Loaded the weights of last epoch {} with score={}'.format(
                self.state.recorded_epochs, self.state.scores[-1]
            ))
            if not self.is_early_stopping() and start_epoch < end_epoch:
                self.logger.info('Now calculating validation score (to verify the restoring success)')
                valid_score, = self.calculate_validation_scores(sess, model, data, [self.scorer])
                self.logger.info('Score={:.4f}'.format(valid_score))

        self.logger.info('Running from epoch {} to epoch {}'.format(start_epoch, end_epoch - 1))

        for epoch in range(start_epoch, end_epoch):
            if self.is_early_stopping():
                self.logger.info('Early stopping (no improvement in the last {} epochs)'.format(
                    self.state.recorded_epochs - self.state.best_epoch
                ))
                break

            self.logger.info('Epoch {}/{}'.format(epoch, self.n_epochs))

            self.logger.debug('Preparing epoch')
            self.prepare_next_epoch(model, data, sess, epoch)

            bar = _create_progress_bar('loss')
            train_losses = []  # used to calculate the epoch train loss
            recent_train_losses = []  # used to calculate the display loss

            self.logger.debug('Training')
            for _ in bar(range(self.get_n_batches())):
                self.global_step += self.batchsize
                train_questions, train_answers_good, train_answers_bad = self.get_next_batch(model, data, sess)

                _, loss, loss_individual, summary = sess.run(
                    [train, model.loss, model.loss_individual, model.summary],
                    feed_dict={
                        learning_rate: self.epoch_learning_rate,
                        model.input_question: train_questions,
                        model.input_answer_good: train_answers_good,
                        model.input_answer_bad: train_answers_bad,
                        model.dropout_keep_prob: self.dropout_keep_prob
                    })
                recent_train_losses = ([loss] + recent_train_losses)[:20]
                train_losses.append(loss)
                bar.dynamic_messages['loss'] = np.mean(recent_train_losses)
                self.add_summary(summary)
            self.logger.info('train loss={:.6f}'.format(np.mean(train_losses)))

            self.logger.info('Now calculating validation score')
            scorers = [self.scorer] + self.scorer_print
            valid_scores = self.calculate_validation_scores(sess, model, data, scorers)
            for scorer, score in zip(scorers, valid_scores):
                self.logger.info('{}={:.4f}'.format(scorer, score))
            valid_score = valid_scores[0]

            # if the validation score is better than the best observed previous loss, create a checkpoint
            self.record_epoch(sess, valid_score)

        if self.state.best_epoch < self.state.recorded_epochs:
            self.logger.info('Restoring the weights of the best epoch {} with score {}'.format(
                self.state.best_epoch, self.state.best_score
            ))
            self.restore_best_weights(sess)

            if self.scorer:
                self.logger.info('Now calculating validation score (to verify the restoring success)')
                valid_score, = self.calculate_validation_scores(sess, model, data, [self.scorer])
                self.logger.info('Score={:.4f}'.format(valid_score))

        return self.state.best_epoch, self.state.best_score

    def calculate_valid_loss(self, sess, model, data):
        if self._valid_questions is None:
            self.logger.debug('Creating validation data')
            self._valid_questions, self._valid_answers_good, self._valid_answers_bad = data.get_items(
                data.archive.valid.qa,
                negative_answers=self.negative_answers_valid
            )

        losses = []
        bar = _create_progress_bar('loss')
        for batch in bar(range(int(math.ceil(len(self._valid_questions) / float(self.batchsize_valid))))):
            batch_start_idx = batch * self.batchsize_valid
            batch_questions = self._valid_questions[batch_start_idx: batch_start_idx + self.batchsize_valid]
            batch_answers_good = self._valid_answers_good[batch_start_idx: batch_start_idx + self.batchsize_valid]
            batch_answers_bad = self._valid_answers_bad[batch_start_idx: batch_start_idx + self.batchsize_valid]

            loss_individual = sess.run(
                model.loss_individual,
                feed_dict={
                    model.input_question: batch_questions,
                    model.input_answer_good: batch_answers_good,
                    model.input_answer_bad: batch_answers_bad,
                    model.dropout_keep_prob: 1.0
                })

            losses += loss_individual.tolist()
            bar.dynamic_messages['loss'] = np.mean(losses)

        return np.mean(losses)

    def prepare_next_epoch(self, model, data, sess, epoch):
        """Prepares the next epoch, especially the batches"""
        self.epoch_learning_rate = self.initial_learning_rate
        if self.dynamic_learning_rate:
            self.epoch_learning_rate /= float(epoch)

    def get_n_batches(self):
        raise NotImplementedError()

    def get_next_batch(self, model, data, sess):
        """Return the training data for the next batch

        :return: questions, good answers, bad answers
        :rtype: list, list, list
        """
        pass


class QATrainState(object):
    def __init__(self, path, less_is_better, logger):
        """Represents the a training state

        :param path: the folder where the checkpoints should be written to
        :param less_is_better: True if a smaller validation score is desired
        """
        self.path = path
        self.logger = logger
        self.less_is_better = less_is_better
        self._saver = None

        self.initialize()

    def initialize(self):
        self.scores = []
        self.best_score = -1 if not self.less_is_better else 2
        self.best_epoch = 0
        self.recorded_epochs = 0
        if self.path and not os.path.exists(self.path):
            os.mkdir(self.path)

    def load(self, session, weights='last'):
        """

        :param session:
        :param weights: 'last' or 'best'
        :return:
        """
        if os.path.exists(self.scores_file):
            scores = []
            with open(self.scores_file, 'r') as f:
                for line in f:
                    scores.append(float(line))

            self.scores = scores
            op = max if not self.less_is_better else min
            self.best_score = op(scores)
            self.best_epoch = scores.index(self.best_score) + 1
            self.recorded_epochs = len(scores)

        restore_path = '{}-{}'.format(
            self.checkpoint_file,
            self.recorded_epochs if weights == 'last' else self.best_epoch
        )
        if os.path.exists(restore_path) or os.path.exists('{}.index'.format(restore_path)):
            self.saver.restore(session, restore_path)
        else:
            self.logger.info('Could not restore weights. Path does not exist: {}'.format(restore_path))

    def record(self, session, score):
        self.recorded_epochs += 1
        self.scores.append(score)
        with open(self.scores_file, 'a') as f:
            f.write('{}\n'.format(score))
        self.saver.save(session, self.checkpoint_file, global_step=self.recorded_epochs)

        if (not self.less_is_better and score > self.best_score) or (self.less_is_better and score < self.best_score):
            self.best_score = score
            self.best_epoch = self.recorded_epochs

    def clear(self):
        shutil.rmtree(self.path)
        self._saver = None
        self.initialize()

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=None)
        return self._saver

    @property
    def scores_file(self):
        return os.path.join(self.path, 'scores.txt')

    @property
    def checkpoint_file(self):
        return os.path.join(self.path, 'model-checkpoint')


def _create_progress_bar(dynamic_msg=None):
    widgets = [
        ' [batch ', progressbar.SimpleProgress(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') '
    ]
    if dynamic_msg is not None:
        widgets.append(progressbar.DynamicMessage(dynamic_msg))
    return progressbar.ProgressBar(widgets=widgets)
