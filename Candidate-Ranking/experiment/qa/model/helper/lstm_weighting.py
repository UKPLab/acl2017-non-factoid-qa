import tensorflow as tf
from tensorflow.python.ops import rnn_cell

from experiment import ComponentBase
from experiment.qa.model import weight_variable
from experiment.qa.model.helper.pooling_helper import non_zero_tokens, attention_softmax


class LSTMBasedImportanceWeighting(ComponentBase):
    def __init__(self, config, config_global, logger):
        super(LSTMBasedImportanceWeighting, self).__init__(config, config_global, logger)
        self.__lstm_history = set()
        self.only_reduction = self.config.get('only_reduction', True)

    @property
    def lstm_pooling_cell_size(self):
        return self.config.get('lstm_pooling_cell_size', 50)

    def importance_weighting(self, raw_representation, indices, item_type, apply_softmax=True):
        re_use = item_type in self.__lstm_history
        self.__lstm_history.add(item_type)

        if item_type == 'question':
            lstm_cell_fw = self.lstm_cell_weighting_Q_fw
            lstm_cell_bw = self.lstm_cell_weighting_Q_bw
            mul_weight = self.mul_Q
            reduction_weight = self.reduction_Q
        else:
            lstm_cell_fw = self.lstm_cell_weighting_A_fw
            lstm_cell_bw = self.lstm_cell_weighting_A_bw
            mul_weight = self.mul_A
            reduction_weight = self.reduction_A

        tensor_non_zero_token = non_zero_tokens(tf.to_float(indices))
        sequence_length = tf.to_int64(tf.reduce_sum(tensor_non_zero_token, 1))

        with tf.variable_scope('positional_weighting_lstm_{}'.format(item_type), reuse=re_use):
            lstm_outputs, _last = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell_fw,
                lstm_cell_bw,
                raw_representation,
                dtype=tf.float32,
                sequence_length=sequence_length
            )

            lstm_output = tf.concat(axis=2, values=lstm_outputs)

        # apply dense over each individual lstm output
        if self.only_reduction:
            flat_lstm_output = tf.reshape(lstm_output, [-1, self.lstm_pooling_cell_size * 2])
            dense_mul_flat = tf.matmul(flat_lstm_output, reduction_weight)
            h1_layer = tf.reshape(dense_mul_flat, [-1, tf.shape(raw_representation)[1]])
        else:
            flat_lstm_output = tf.reshape(raw_representation, [-1, self.lstm_pooling_cell_size * 2])
            dense_mul_flat = tf.nn.tanh(tf.matmul(flat_lstm_output, mul_weight))
            reduction = tf.matmul(dense_mul_flat, reduction_weight)
            h1_layer = tf.reshape(reduction, [-1, tf.shape(raw_representation)[1]])

        if apply_softmax:
            return attention_softmax(h1_layer, tensor_non_zero_token)
        else:
            return h1_layer

    def initialize_weights(self):
        cell_size = self.lstm_pooling_cell_size
        self.mul_Q = weight_variable('mul_Q', [cell_size * 2, cell_size * 2])
        self.reduction_Q = weight_variable('reduction_Q', [cell_size * 2, 1])
        self.mul_A = weight_variable('mul_A', [cell_size * 2, cell_size * 2])
        self.reduction_A = weight_variable('reduction_A', [cell_size * 2, 1])

        with tf.variable_scope('lstm_cell_weighting_Q_fw'):
            self.lstm_cell_weighting_Q_fw = rnn_cell.BasicLSTMCell(cell_size, state_is_tuple=True)

        with tf.variable_scope('lstm_cell_weighting_Q_bw'):
            self.lstm_cell_weighting_Q_bw = rnn_cell.BasicLSTMCell(cell_size, state_is_tuple=True)

        with tf.variable_scope('lstm_cell_weighting_A_fw'):
            self.lstm_cell_weighting_A_fw = rnn_cell.BasicLSTMCell(cell_size, state_is_tuple=True)

        with tf.variable_scope('lstm_cell_weighting_A_bw'):
            self.lstm_cell_weighting_A_bw = rnn_cell.BasicLSTMCell(cell_size, state_is_tuple=True)


