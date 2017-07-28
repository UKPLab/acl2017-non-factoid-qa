# coding: utf8
from experiment.qa.model.bilstm import BiLSTMModel
from experiment.qa.model.helper.lstm_weighting import LSTMBasedImportanceWeighting
from experiment.qa.model.helper.pooling_helper import weighted_pooling


class LWBiLSTMModel(BiLSTMModel, LSTMBasedImportanceWeighting):
    def __init__(self, config, config_global, logger):
        """This is the model proposed in

            Rücklé, Andreas, and Iryna Gurevych. 2017.
            “Representation Learning for Answer Selection with LSTM-Based Importance Weighting.”
            In Proceedings of the 12th International Conference on Computational Semantics (IWCS 2017),
            (to appear).

        It uses a separate BiLSTM to determine the importance of segments in the input. This is similar
        to self-attention.

        :param config:
        :param config_global:
        :param logger:
        """
        super(LWBiLSTMModel, self).__init__(config, config_global, logger)
        self.shared_lstm_pooling = self.config.get('shared_lstm_pooling', False)

    def build(self, data, sess):
        self.build_input(data, sess)
        self.initialize_weights()

        raw_representation_question = self.bilstm_representation_raw(
            self.embeddings_question, self.input_question, False
        )
        raw_representation_answer_good = self.bilstm_representation_raw(
            self.embeddings_answer_good, self.input_answer_good, True
        )
        raw_representation_answer_bad = self.bilstm_representation_raw(
            self.embeddings_answer_bad, self.input_answer_bad, True
        )

        self.question_importance_weight = self.importance_weighting(
            raw_representation_question,
            self.input_question,
            item_type='question' if not self.shared_lstm_pooling else 'shared'
        )
        self.answer_good_importance_weight = self.importance_weighting(
            raw_representation_answer_good,
            self.input_answer_good,
            item_type='answer' if not self.shared_lstm_pooling else 'shared'
        )
        self.answer_bad_importance_weight = self.importance_weighting(
            raw_representation_answer_bad,
            self.input_answer_bad,
            item_type='answer' if not self.shared_lstm_pooling else 'shared'
        )

        pooled_representation_question = weighted_pooling(
            raw_representation_question, self.question_importance_weight, self.input_question
        )
        pooled_representation_answer_good = weighted_pooling(
            raw_representation_answer_good, self.answer_good_importance_weight, self.input_answer_good
        )
        pooled_representation_answer_bad = weighted_pooling(
            raw_representation_answer_bad, self.answer_bad_importance_weight, self.input_answer_bad
        )

        self.create_outputs(
            pooled_representation_question,
            pooled_representation_answer_good,
            pooled_representation_question,
            pooled_representation_answer_bad
        )

    def initialize_weights(self):
        """Global initialization of weights for the representation layer

        """
        BiLSTMModel.initialize_weights(self)
        LSTMBasedImportanceWeighting.initialize_weights(self)


component = LWBiLSTMModel
