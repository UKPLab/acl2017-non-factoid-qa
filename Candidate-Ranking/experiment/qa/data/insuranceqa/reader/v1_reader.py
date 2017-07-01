from collections import OrderedDict

from experiment.qa.data.models import *
from experiment.qa.data.reader import TSVArchiveReader
import numpy as np

class V1Reader(TSVArchiveReader):
    def file_path(self, filename):
        return '{}/V1/{}'.format(self.archive_path, filename)

    def read_split(self, name, vocab, answers):
        answers_list = list(answers.values())
        train = name == 'train'
        filename = 'question.train.token_idx.label' if train else 'question.{}.label.token_idx.pool'.format(name)
        datapoints = []
        for i, line in enumerate(self.read_tsv(self.file_path(filename))):
            question_line = line[0] if train else line[1]
            ground_truth_line = line[1] if train else line[0]
            pool_line = None if train else line[2]

            ground_truth = [answers[gt] for gt in ground_truth_line.split(' ')]
            if pool_line:
                pool = [answers[pa] for pa in pool_line.split(' ')]
                np.random.shuffle(pool)
            else:
                # if there is no pool line (in training), we use all answers as the pool
                pool = answers_list

            question_tokens = [Token(vocab[t]) for t in question_line.split()]
            question_sentence = Sentence(' '.join([t.text for t in question_tokens]), question_tokens)
            question = TextItem(question_sentence.text, [question_sentence])

            datapoints.append(QAPool(question, pool, ground_truth))

        return Data(name, datapoints, list(answers.values()))

    def read(self):
        vocab = dict(self.read_tsv(self.file_path('vocabulary')))
        answers = OrderedDict()
        for line in self.read_tsv(self.file_path('answers.label.token_idx')):
            id = line[0]
            tokens = [Token(vocab[t]) for t in line[1].split(' ')]
            answer_sentence = Sentence(' '.join(t.text for t in tokens), tokens)
            answer = TextItem(answer_sentence.text, [answer_sentence])
            answer.metadata['id'] = id
            answers[id] = answer

        train = self.read_split("train", vocab, answers)
        valid = self.read_split("dev", vocab, answers)
        test1 = self.read_split("test1", vocab, answers)
        test2 = self.read_split("test2", vocab, answers)

        questions = [qa.question for qa in train.qa + valid.qa + test1.qa + test2.qa]

        return Archive(train, valid, [test1, test2], questions, list(answers.values()))
