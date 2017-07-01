from __future__ import division

import math
import os

import numpy as np
from unidecode import unidecode

import experiment


class QAEvaluation(experiment.Evaluation):
    def __init__(self, config, config_global, logger):
        super(QAEvaluation, self).__init__(config, config_global, logger)
        self.batchsize_test = self.config.get('batchsize_test', 50)

    def start(self, model, data, sess):
        length_question = self.config_global['question_length']
        length_answer = self.config_global['answer_length']

        evaluation_data = data.archive.test
        if self.config.get('include_valid', False):
            evaluation_data = [data.archive.valid] + evaluation_data

        output_path = self.config.get('output', None)
        if output_path and not os.path.exists(output_path):
            os.mkdir(output_path)

        for split in evaluation_data:
            self.logger.info("Evaluating {}".format(split.split_name))
            ranks = []
            average_precisions = []

            for pool in split.qa:
                test_questions = np.array(
                    [data.get_item_vector(pool.question, length_question)] * len(pool.pooled_answers)
                )
                test_answers = np.array(
                    [data.get_item_vector(a, length_answer) for a in pool.pooled_answers]
                )

                scores = []
                for test_batch in range(int(math.ceil(len(test_answers) / float(self.batchsize_test)))):
                    test_batch_indices = self.batchsize_test * test_batch, self.batchsize_test * (test_batch + 1)
                    test_batch_scores, = sess.run([model.predict], feed_dict={
                        model.input_question: test_questions[test_batch_indices[0]:test_batch_indices[1]],
                        model.input_answer_good: test_answers[test_batch_indices[0]:test_batch_indices[1]],
                        model.dropout_keep_prob: 1.0,
                    })
                    scores += test_batch_scores.tolist()

                sorted_answers = sorted(zip(scores, pool.pooled_answers), key=lambda x: -x[0])
                rank = 0
                precisions = []
                for i, (score, answer) in enumerate(sorted_answers, start=1):
                    if answer in pool.ground_truth:
                        if rank == 0:
                            rank = i
                        precisions.append((len(precisions) + 1) / float(i))

                ranks.append(rank)
                average_precisions.append(np.mean(precisions))
                self.logger.debug('Rank: {}'.format(rank))

            correct_answers = len([a for a in ranks if a == 1])
            accuracy = correct_answers / float(len(ranks))
            mrr = np.mean([1 / float(r) for r in ranks])
            map = np.mean(average_precisions)

            self.logger.info('Correct answers: {}/{}'.format(correct_answers, len(split.qa)))
            self.logger.info('Accuracy: {}'.format(accuracy))
            self.logger.info('MRR: {}'.format(mrr))
            self.logger.info('MAP: {}'.format(map))


component = QAEvaluation
