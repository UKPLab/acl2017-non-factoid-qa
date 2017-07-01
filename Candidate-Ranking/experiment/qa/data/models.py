from experiment.util import unique_items


class MetadataItem(object):
    def __init__(self):
        self.metadata = dict()


class Token(MetadataItem):
    def __init__(self, text):
        """

        :type text: str
        """
        super(Token, self).__init__()
        self.text = text


class Sentence(MetadataItem):
    def __init__(self, text, tokens):
        """

        :type text: str
        :type tokens: list[Token]
        """
        super(Sentence, self).__init__()
        self.text = text
        self.tokens = tokens

    @property
    def vocab(self):
        vocab = []
        for token in self.tokens:
            vocab.append(token.text)
        return unique_items(vocab)


class TextItem(MetadataItem):
    def __init__(self, text, sentences):
        """

        :type text: str
        :type sentences: list[Sentence]
        """
        super(TextItem, self).__init__()
        self.text = text
        self.sentences = sentences

    @property
    def vocab(self):
        vocab = []
        for sentence in self.sentences:
            vocab += sentence.vocab
        return unique_items(vocab)


class QAPool(object):
    def __init__(self, question, pooled_answers, ground_truth):
        """

        :type question: TextItem
        :type pooled_answers: list[TextItem]
        :type ground_truth: list[TextItem]
        """
        self.question = question
        self.pooled_answers = pooled_answers
        self.ground_truth = ground_truth


class Data(object):
    def __init__(self, split_name, qa, answers):
        """

        :type split_name: str
        :type qa: list[QAPool]
        :type answers: list[TextItem]
        """
        self.split_name = split_name
        self.qa = qa
        self.answers = answers


class Archive(object):
    def __init__(self, train, valid, test, questions, answers):
        """

        :type train: Data
        :type valid: Data
        :type test: list[Data]
        :type questions: list[TexItem]
        :type answers: list[TexItem]
        """
        self.train = train
        self.valid = valid
        self.test = test
        self.questions = questions
        self.answers = answers

        self._vocab = None  # lazily created

    @property
    def vocab(self):
        """
        :rtype: set
        """
        if self._vocab is None:
            self._vocab = []
            for question in self.questions:
                self._vocab += question.vocab
            for answer in self.answers:
                self._vocab += answer.vocab
            self._vocab = unique_items(self._vocab)

        return self._vocab
