# coding=utf-8

import math

import numpy as np

from experiment.qa.train import QABatchedTraining


class InsuranceQATrainingDynamic(QABatchedTraining):
    """This is a training method that tries to replicate training method used in recent work on QA (esp. insuranceqa):

    For each question, we randomly sample a list of N (e.g. 50) negative answers and choose the negative answer which
    has the highest similarity to the question. This is then used to compute the training loss and update gradients.
    """

    def __init__(self, config, config_global, logger):
        super(InsuranceQATrainingDynamic, self).__init__(config, config_global, logger)
        self.batch_i = 0
        self.examples_incomplete = []

    def prepare_next_epoch(self, model, data, sess, epoch):
        super(InsuranceQATrainingDynamic, self).prepare_next_epoch(model, data, sess, epoch)
        self.batch_i = 0

        # we prepare all questions and good answers in advance. Bad answers will be pooled at run-time
        if len(self.examples_incomplete) == 0:
            self.logger.debug('Preparing training examples (incomplete)')
            self.examples_incomplete = []
            for pool in data.archive.train.qa:
                question_vec = data.get_item_vector(pool.question, self.length_question)

                # we allow the ground-truth to be either all correct answers or only the first correct answer (which
                # might be the best one)
                ground_truth = pool.ground_truth
                if self.config.get('only_first_ground_truth', False):
                    ground_truth = ground_truth[:1]

                ground_truth_vecs = [data.get_item_vector(ga, self.length_answer) for ga in ground_truth]
                negative_answers_pool = data.archive.train.answers
                self.examples_incomplete.append(
                    (question_vec, ground_truth_vecs, negative_answers_pool, ground_truth)
                )

        # shuffle the indices of each batch
        self.epoch_random_indices = np.random.permutation(len(self.examples_incomplete))

    def get_n_batches(self):
        return int(math.ceil(len(self.examples_incomplete) / float(self.batchsize)))

    def get_next_batch(self, model, data, sess):
        """We calculate the training examples as follows: for each of the questions and associated good answers, we
        are fetching 50 random negative answers. For each of those answers, we calculate the similarity to the
        question using the model prediction. We use the negative answer with the highest similarity to construct
        triples for training.

        :return: questions, good answers, bad answers
        :rtype: list, list, list
        """
        batch_questions = []
        batch_answers_good = []
        batch_answers_bad = []

        indices = self.epoch_random_indices[self.batch_i * self.batchsize: (self.batch_i + 1) * self.batchsize]
        incomplete_data_epoch = [self.examples_incomplete[i] for i in indices]

        # we create all the data that is required to run the predictions for this batch. We bundle all predictions
        # because of computational efficiency
        data_info = []    # (question, ground_truth, answers_bad, prediction_start_index)
        prediction_questions = []
        prediction_answers_bad = []
        prediction_results = []
        for question, ground_truth_vecs, pool, ground_truth in incomplete_data_epoch:
            begin_index = len(prediction_questions)
            negative_answer_indices = np.random.random_integers(0, len(pool) - 1, self.negative_answers)
            answers_bad = [pool[i] for i in negative_answer_indices if pool[i] not in ground_truth]
            if len(answers_bad) > 0:
                answers_bad_vecs = [data.get_item_vector(a, self.length_answer) for a in answers_bad]
                prediction_questions += [question] * len(answers_bad)
                prediction_answers_bad += answers_bad_vecs
                data_info.append((question, ground_truth_vecs, answers_bad_vecs, begin_index))

        if self.negative_answers > 1:
            # we now run all the predictions in batched mode, which is significantly faster than running a prediction
            # for each individual question
            for predict_batch in range(int(math.ceil(len(prediction_questions) / float(self.batchsize_valid)))):
                batch_start_idx = predict_batch * self.batchsize_valid
                predict_batch_questions = prediction_questions[batch_start_idx: batch_start_idx + self.batchsize_valid]
                predict_batch_answers = prediction_answers_bad[batch_start_idx: batch_start_idx + self.batchsize_valid]

                predictions, = sess.run([model.predict], feed_dict={
                    model.input_question: predict_batch_questions,
                    model.input_answer_good: predict_batch_answers,
                    model.dropout_keep_prob: 1.0
                })
                prediction_results += list(predictions)
        else:
            prediction_results = [1.0] * len(data_info)

        # now we are processing all the predictions and generate the batch data
        for question, ground_truth_vecs, answers_bad, prediction_start_index in data_info:
            predictions_question = prediction_results[prediction_start_index:prediction_start_index + len(answers_bad)]
            most_similar_answer_bad_vector = answers_bad[np.argmax(predictions_question)]

            batch_questions.append(question)
            batch_answers_bad.append(most_similar_answer_bad_vector)

            # we choose only one of the good answers to not over-train on the questions which have more good answers
            # than others
            batch_answers_good.append(ground_truth_vecs[self.state.recorded_epochs % len(ground_truth_vecs)])

        self.batch_i += 1
        return batch_questions, batch_answers_good, batch_answers_bad


component = InsuranceQATrainingDynamic
