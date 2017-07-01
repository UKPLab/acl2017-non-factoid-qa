import tensorflow as tf
from tensorflow.python.ops import rnn_cell

from experiment.qa.model import QAModel
from experiment.qa.model.helper.pooling_helper import non_zero_tokens, maxpool


class BiLSTMModel(QAModel):
    """An LSTM model with 1-max pooling to learn the representation"""
    def __init__(self, config, config_global, logger):
        super(BiLSTMModel, self).__init__(config, config_global, logger)
        self.lstm_cell_size = self.config['lstm_cell_size']

    def build(self, data, sess):
        self.build_input(data, sess)

        # we initialize the weights of the representation layers globally so that they can be applied to both, questions
        # and (good/bad)answers. This is an important part, otherwise results would be much worse.
        self.initialize_weights()

        representation_question = maxpool(
            self.bilstm_representation_raw(
                self.embeddings_question,
                self.input_question,
                re_use_lstm=False
            )
        )
        representation_answer_good = maxpool(
            self.bilstm_representation_raw(
                self.embeddings_answer_good,
                self.input_answer_good,
                re_use_lstm=True
            )
        )
        representation_answer_bad = maxpool(
            self.bilstm_representation_raw(
                self.embeddings_answer_bad,
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
        with tf.variable_scope('lstm_cell_fw'):
            self.lstm_cell_forward = rnn_cell.BasicLSTMCell(self.lstm_cell_size, state_is_tuple=True)
        with tf.variable_scope('lstm_cell_bw'):
            self.lstm_cell_backward = rnn_cell.BasicLSTMCell(self.lstm_cell_size, state_is_tuple=True)

    def bilstm_representation_raw(self, item, indices, re_use_lstm, name='lstm'):
        """Creates a representation graph which retrieves a text item (represented by its word embeddings) and returns
        a vector-representation

        :param item: the text item. Can be question or (good/bad) answer
        :param sequence_length: maximum length of the text item
        :param re_use_lstm: should be False for the first call, True for al subsequent ones to get the same lstm
        variables
        :return: representation tensor
        """
        tensor_non_zero_token = non_zero_tokens(tf.to_float(indices))
        sequence_length = tf.to_int64(tf.reduce_sum(tensor_non_zero_token, 1))

        with tf.variable_scope(name, reuse=re_use_lstm):
            output, _last = tf.nn.bidirectional_dynamic_rnn(
                self.lstm_cell_forward,
                self.lstm_cell_backward,
                item,
                dtype=tf.float32,
                sequence_length=sequence_length
            )

            return tf.concat(axis=2, values=output)


component = BiLSTMModel
