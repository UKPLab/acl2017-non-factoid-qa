import numpy as np

from experiment.qa.data.models import *
from experiment.qa.data.reader import TSVArchiveReader


class V2Reader(TSVArchiveReader):
    def __init__(self, archive_path, lowercased, logger, pooled_answers=500, tokenizer='token'):
        super(V2Reader, self).__init__(archive_path, lowercased, logger)
        self.pooled_answers = pooled_answers
        self.tokenizer = tokenizer

    def file_path(self, filename):
        return '{}/V2/{}'.format(self.archive_path, filename)

    def read_split(self, name, vocab, answers):
        filename = 'InsuranceQA.question.anslabel.{}.{}.pool.solr.{}.encoded.gz'.format(
            self.tokenizer, self.pooled_answers, name
        )
        datapoints = []
        split_answers = []
        for i, line in enumerate(self.read_tsv(self.file_path(filename), is_gzip=True)):
            question_tokens = [Token(vocab[t]) for t in line[1].split()]
            question_sentence = Sentence(' '.join([t.text for t in question_tokens]), question_tokens)
            question = TextItem(question_sentence.text, [question_sentence])
            question.metadata['id'] = '{}-{}'.format(name, i)

            ground_truth = [answers[gt] for gt in line[2].split(' ')]
            pool = [answers[pa] for pa in line[3].split(' ')]
            np.random.shuffle(pool)
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
        vocab = dict(self.read_tsv(self.file_path('vocabulary')))
        answers_path = 'InsuranceQA.label2answer.{}.encoded.gz'.format(self.tokenizer)
        answers = dict()
        for line in self.read_tsv(self.file_path(answers_path), is_gzip=True):
            id = line[0]
            tokens = [Token(vocab[t]) for t in line[1].split(' ')]
            answer_sentence = Sentence(' '.join(t.text for t in tokens), tokens)
            answer = TextItem(answer_sentence.text, [answer_sentence])
            answer.metadata['id'] = id
            answers[id] = answer

        train = self.read_split("train", vocab, answers)
        valid = self.read_split("valid", vocab, answers)
        test = self.read_split("test", vocab, answers)

        questions = [qa.question for qa in (train.qa + valid.qa + test.qa)]

        return Archive(train, valid, [test], questions, list(answers.values()))
