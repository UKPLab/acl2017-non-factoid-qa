import math

import numpy as np

import experiment
from experiment.qa.data import models
from experiment.util import read_embeddings


class QAData(experiment.Data):
    def __init__(self, config, config_global, logger):
        super(QAData, self).__init__(config, config_global, logger)

        # public fields
        self.archive = None  # Archive
        self.vocab_to_index = dict()  # a dict that matches each token to an integer position for the embeddings
        self.embeddings = None  # numpy array that contains all relevant embeddings
        self.lowercased = self.config.get('lowercased', False)

        self.map_oov = self.config.get('map_oov', False)
        self.map_numbers = self.config.get('map_numbers', False)

    def _get_reader(self):
        """:rtype: InsuranceQAReader"""
        raise NotImplementedError()

    def setup(self):
        self.logger.info('Loading dataset')
        reader = self._get_reader()
        self.archive = reader.read()

        self.logger.debug('Dataset questions: train={}, dev={}, test={}'.format(
            len(self.archive.train.qa),
            len(self.archive.valid.qa),
            [len(t.qa) for t in self.archive.test]
        ))
        # self.logger.debug('Answer length: mean={}, median={}'.format(
        #     np.mean([len(a.sentences[0].tokens) for a in self.archive.answers]),
        #     np.median([len(a.sentences[0].tokens) for a in self.archive.answers])
        # ))
        qas = self.archive.train.qa + self.archive.valid.qa
        for t in self.archive.test:
            qas += t.qa
        self.logger.debug('Mean answer count={}'.format(
            np.mean([len(p.ground_truth) for p in qas])
        ))
        self.logger.debug('Mean poolsize={}'.format(
            np.mean([len(p.pooled_answers) for p in qas])
        ))

        embedding_size = self.config_global['embedding_size']
        if 'embeddings_path' in self.config:
            # load the initial embeddings
            self.logger.info('Fetching the dataset vocab')
            vocab = self.archive.vocab
            self.logger.info('Loading embeddings (vocab size={})'.format(len(vocab)))
            embeddings_dict = read_embeddings(self.config['embeddings_path'], vocab, self.logger)

            zero_padding = np.zeros((embedding_size,))
            number = np.random.uniform(-1.0, 1.0, [embedding_size, ])
            oov = np.random.uniform(-1.0, 1.0, [embedding_size, ])
            embeddings = [zero_padding, oov, number]

            n_oov = 0
            for token in self.archive.vocab:
                embedding_dict_item = embeddings_dict.get(token, None)
                digits = sum([1 if char.isdigit() else 0 for char in token])
                is_number = digits > len(token) / 2.0

                if is_number and self.map_numbers:
                    self.vocab_to_index[token] = 2  # number
                elif embedding_dict_item is not None:
                    self.vocab_to_index[token] = len(embeddings)
                    embeddings.append(embedding_dict_item)
                else:
                    n_oov += 1
                    if self.map_oov:
                        self.vocab_to_index[token] = 1  # oov
                    else:
                        # for each oov, we create a new random vector
                        self.vocab_to_index[token] = len(embeddings)
                        embeddings.append(np.random.uniform(-1.0, 1.0, [embedding_size, ]))

            self.embeddings = np.array(embeddings)
            self.logger.info('OOV tokens: {}'.format(n_oov))

        else:
            self.vocab_to_index = dict([(t, i) for (i, t) in enumerate(self.archive.vocab, start=1)])
            self.embeddings = np.append(
                np.zeros((1, embedding_size)),  # zero-padding
                np.random.uniform(-1.0, 1.0, [len(self.archive.vocab), embedding_size]),
                axis=0
            )

    def get_fold_data(self, fold_i, n_folds):
        """The fold data is created as follows. We concatenate train and valid data and split it into n_folds equal
        parts. The fold_ith part is the validation split, all others are the training data.

        :param fold_i: number of the fold (starting at zero)
        :param n_folds: number of total folds
        :return: a new instance of QAData
        """
        train_valid = self.archive.train.qa + self.archive.valid.qa
        splits = []

        fold_size = math.floor(len(train_valid) / n_folds)
        for i in range(n_folds):
            splits.append(train_valid[i * fold_size:(i + 1) * fold_size])

        splits_train = [s for (i, s) in enumerate(splits) if i != fold_i]
        qa_train = [pool for split in splits_train for pool in split]
        answers_train = [a for p in qa_train for a in p.pooled_answers]
        train_data = models.Data('train', qa_train, answers=answers_train)
        qa_valid = splits[fold_i]
        answers_valid = [a for p in qa_valid for a in p.pooled_answers]
        valid_data = models.Data('valid', qa_valid, answers=answers_valid)

        archive_fold = models.Archive(
            train_data, valid_data, self.archive.test, self.archive.questions, self.archive.answers
        )

        data_fold = QAData(self.config, self.config_global, self.logger)
        data_fold.archive = archive_fold
        data_fold.vocab_to_index = self.vocab_to_index
        data_fold.embeddings = self.embeddings
        return data_fold

    def get_item_vector(self, ti, max_len):
        tokens = []
        for sentence in ti.sentences:
            if len(tokens) < max_len:
                # 1 = oov
                tokens += [self.vocab_to_index[t.text] if t.text in self.vocab_to_index else 1 for t in sentence.tokens]
        tokens = tokens[:max_len]

        # zero-pad to max_len
        tokens_padded = tokens + [0 for _ in range(max_len - len(tokens))]
        return tokens_padded

    def get_items(self, qa, negative_answers=None):
        """Returns randomly constructed samples for all questions inside the qa list with a specific number of negative
        answers

        :param qa:
        :param negative_answers: None=all
        :return:
        """
        questions = []
        answers_good = []
        answers_bad = []

        length_question = self.config_global['question_length']
        length_answer = self.config_global['answer_length']

        for pool in qa:
            question = self.get_item_vector(pool.question, length_question)
            for answer_good_item in pool.ground_truth:
                answer_good = self.get_item_vector(answer_good_item, length_answer)
                shuffled_pool = [pa for pa in pool.pooled_answers if pa not in pool.ground_truth]
                np.random.shuffle(shuffled_pool)
                for answer_bad_item in shuffled_pool[:negative_answers]:
                    answer_bad = self.get_item_vector(answer_bad_item, length_answer)
                    questions.append(question)
                    answers_good.append(answer_good)
                    answers_bad.append(answer_bad)

        questions = np.array(questions, dtype=np.int32)
        answers_good = np.array(answers_good, dtype=np.int32)
        answers_bad = np.array(answers_bad, dtype=np.int32)

        return questions, answers_good, answers_bad
