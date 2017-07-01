import numpy as np

from experiment.qa.data.models import *
from experiment.qa.data.reader import TSVArchiveReader
import os


class TSVReader(TSVArchiveReader):
    def __init__(self, archive_path, lowercased, logger):
        super(TSVReader, self).__init__(archive_path, lowercased, logger)

    def file_path(self, filename):
        return os.path.join(self.archive_path, filename)

    def read_items(self, name, vocab):
        items_path = self.file_path('{}.tsv.gz'.format(name))
        items = dict()
        for line in self.read_tsv(items_path, is_gzip=True):
            id = line[0]
            text = line[1] if len(line) > 1 else ''
            tokens = [Token(vocab[t]) for t in text.split()]
            answer_sentence = Sentence(' '.join(t.text for t in tokens), tokens)
            answer = TextItem(answer_sentence.text, [answer_sentence])
            answer.metadata['id'] = id
            items[id] = answer
        return items

    def read_split(self, name, questions, answers):
        split_path = self.file_path('{}.tsv.gz'.format(name))
        datapoints = []
        split_answers = []
        for i, line in enumerate(self.read_tsv(split_path, is_gzip=True)):
            question = questions[line[0]]
            ground_truth = [answers[gt_id] for gt_id in line[1].split()]
            pool = [answers[pa_id] for pa_id in line[2].split()]
            # np.random.shuffle(pool)
            datapoints.append(QAPool(question, pool, ground_truth))

            split_answers += pool

        # we filter out all pools that do not contain any ground truth answer
        qa_pools_len_before = len(datapoints)
        datapoints = [p for p in datapoints if len([1 for gt in p.ground_truth if gt in p.pooled_answers]) > 0]
        qa_pools_len_after = len(datapoints)
        self.logger.info("Split {} reduced to {} item from {} due to missing ground truth in pool".format(
            name, qa_pools_len_after, qa_pools_len_before
        ))

        return Data(name, datapoints, split_answers)

    def read(self):
        vocab = dict(self.read_tsv(self.file_path('vocab.tsv.gz'), is_gzip=True))

        answers = self.read_items('answers', vocab)
        questions = self.read_items('questions', vocab)

        train = self.read_split("train", questions, answers)
        valid = self.read_split("valid", questions, answers)
        test = self.read_split("test", questions, answers)

        return Archive(train, valid, [test], list(questions.values()), list(answers.values()))
