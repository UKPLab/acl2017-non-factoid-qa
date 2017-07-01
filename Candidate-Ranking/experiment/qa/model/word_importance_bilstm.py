import tensorflow as tf

from experiment.qa.model import weight_variable
from experiment.qa.model.helper.pooling_helper import maxpool
from experiment.qa.model.bilstm import BiLSTMModel


class WordImportanceLSTMModel(BiLSTMModel):
    """This is a trivial extension of the BiLSTM model that learns a weight for each output word. This allows it to
    ignore stopwords and put a higher emphasis on important words."""

    def build(self, data, sess):
        self.build_input(data, sess)
        self.initialize_weights()

        question_importance = tf.nn.sigmoid(multiply_3_2(self.embeddings_question, self.W_importance))
        answer_good_importance = tf.nn.sigmoid(multiply_3_2(self.embeddings_answer_good, self.W_importance))
        answer_bad_importance = tf.nn.sigmoid(multiply_3_2(self.embeddings_answer_bad, self.W_importance))

        self.question_importance_weight = tf.reshape(question_importance, [-1, self.question_length])
        self.answer_good_importance_weight = tf.reshape(answer_good_importance, [-1, self.answer_length])
        self.answer_bad_importance_weight = tf.reshape(answer_bad_importance, [-1, self.answer_length])

        representation_question = maxpool(
            self.bilstm_representation_raw(
                self.embeddings_question * question_importance,
                self.input_question,
                re_use_lstm=False
            )
        )
        representation_answer_good = maxpool(
            self.bilstm_representation_raw(
                self.embeddings_answer_good * answer_good_importance,
                self.input_answer_good,
                re_use_lstm=True
            )
        )
        representation_answer_bad = maxpool(
            self.bilstm_representation_raw(
                self.embeddings_answer_bad * answer_bad_importance,
                self.input_answer_bad,
                re_use_lstm=True
            )
        )

        self.create_outputs(
            representation_question,
            representation_answer_good,
            representation_question,
            representation_answer_bad
        )

    def initialize_weights(self):
        """Global initialization of weights for the representation layer

        """
        super(WordImportanceLSTMModel, self).initialize_weights()
        self.W_importance = weight_variable('W_importance', [self.embedding_size, 1])


def multiply_3_2(x, y, n_items=None, n_values=None, n_output_values=None):
    """Matmuls each 2d matrix in a 3d tensor with a 2d multiplicator

    :param x: 3d input
    :param y: 2d input
    :param n_items: you can explicitly set the shape of the input to enable better debugging in tensorflow
    :return:
    """
    shape_x = tf.shape(x)
    shape_y = tf.shape(y)

    n_items = shape_x[1] if n_items is None else n_items
    n_values = shape_x[2] if n_values is None else n_values
    n_output_values = shape_y[1] if n_output_values is None else n_output_values

    x_2d = tf.reshape(x, [-1, n_values])
    result_2d = tf.matmul(x_2d, y)
    result_3d = tf.reshape(result_2d, [-1, n_items, n_output_values])
    return result_3d


component = WordImportanceLSTMModel
