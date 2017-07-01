import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer, xavier_initializer

from experiment import Model


class QAModel(Model):
    def __init__(self, config, config_global, logger):
        super(QAModel, self).__init__(config, config_global, logger)
        self.__summary = None

        self.trainable_embeddings = self.config.get('trainable_embeddings', True)
        self.regularization_strength = self.config.get('regularization_strength', None)
        self.question_length = self.config_global['question_length']
        self.answer_length = self.config_global['answer_length']
        self.embedding_size = self.config_global['embedding_size']

        self.__loss_max = None
        self.__loss_mean = None

    def build_input(self, data, sess):
        self.input_question = tf.placeholder(tf.int32, [None, self.question_length])
        self.input_answer_good = tf.placeholder(tf.int32, [None, self.answer_length])
        self.input_answer_bad = tf.placeholder(tf.int32, [None, self.answer_length])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embeddings_init = tf.constant_initializer(data.embeddings)
            embeddings_weight = tf.get_variable("embeddings", data.embeddings.shape, dtype=tf.float32,
                                                initializer=embeddings_init,
                                                trainable=self.trainable_embeddings)

            self.embeddings_question = tf.nn.embedding_lookup(embeddings_weight, self.input_question)
            self.embeddings_answer_good = tf.nn.embedding_lookup(embeddings_weight, self.input_answer_good)
            self.embeddings_answer_bad = tf.nn.embedding_lookup(embeddings_weight, self.input_answer_bad)

        # we set the default for all pooling weights: zeros
        self.question_importance_weight = 0.0 * tf.to_float(self.input_question)
        self.answer_good_importance_weight = 0.0 * tf.to_float(self.input_answer_good)
        self.answer_bad_importance_weight = 0.0 * tf.to_float(self.input_answer_bad)

    def create_outputs(self, question_good, answer_good, question_bad, answer_bad):
        self.representation_question_good = question_good
        self.representation_answer_good = answer_good
        self.representation_question_bad = question_bad
        self.representation_answer_bad = answer_bad

        similarity_type = self.config.get('similarity', 'cosine')
        if similarity_type == 'gesd':
            similarity = gesd_similarity
        else:
            # otherwise => cosine
            similarity = cosine_similarity

        # We apply dropout before similarity. This only works when we dropout the same indices in question and answer.
        # Otherwise, the similarity would be heavily biased (in case of angular/cosine distance).
        dropout_multiplicators = tf.nn.dropout(question_good * 0.0 + 1.0, self.dropout_keep_prob)

        question_good_dropout = question_good * dropout_multiplicators
        answer_good_dropout = answer_good * dropout_multiplicators
        question_bad_dropout = question_bad * dropout_multiplicators
        answer_bad_dropout = answer_bad * dropout_multiplicators

        self.similarity_question_answer_good = similarity(
            question_good_dropout,
            answer_good_dropout,
        )
        self.similarity_question_answer_bad = similarity(
            question_bad_dropout,
            answer_bad_dropout,
        )

        self.loss_individual = hinge_loss(
            self.similarity_question_answer_good,
            self.similarity_question_answer_bad,
            self.config['margin']
        )

        reg = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = tf.reduce_mean(self.loss_individual) + reg
        self.predict = self.similarity_question_answer_good

        tf.summary.scalar('Loss', self.loss)

    @property
    def summary(self):
        if self.__summary is None:
            self.__summary = tf.summary.merge_all(key='summaries')
        return self.__summary


def weight_variable(name, shape, regularization=None):
    regularizer = None
    if regularization is not None:
        regularizer = l2_regularizer(1e-5)

    return tf.get_variable(name, shape=shape, initializer=xavier_initializer(), regularizer=regularizer)


def bias_variable(name, shape, value=0.1):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(value))


def cosine_similarity(a, b):
    return tf.div(
        tf.reduce_sum(tf.multiply(a, b), 1),
        tf.multiply(
            tf.sqrt(tf.reduce_sum(tf.square(a), 1)),
            tf.sqrt(tf.reduce_sum(tf.square(b), 1))
        )
    )


def gesd_similarity(a, b):
    a = tf.nn.l2_normalize(a, dim=1)
    b = tf.nn.l2_normalize(b, dim=1)
    euclidean = tf.sqrt(tf.reduce_sum((a - b) ** 2, 1))
    mm = tf.reshape(
        tf.matmul(
            tf.reshape(a, [-1, 1, tf.shape(a)[1]]),
            tf.transpose(
                tf.reshape(b, [-1, 1, tf.shape(a)[1]]),
                [0, 2, 1]
            )
        ),
        [-1]
    )
    sigmoid_dot = tf.exp(-1 * (mm + 1))
    return 1.0 / (1.0 + euclidean) * 1.0 / (1.0 + sigmoid_dot)


def hinge_loss(similarity_good_tensor, similarity_bad_tensor, margin):
    return tf.maximum(
        0.0,
        tf.add(
            tf.subtract(
                margin,
                similarity_good_tensor
            ),
            similarity_bad_tensor
        )
    )
