from experiment.qa.model import weight_variable
from experiment.qa.model.bilstm import BiLSTMModel
from experiment.qa.model.helper.pooling_helper import attentive_pooling_weights, weighted_pooling


class AttentivePoolingLSTMModel(BiLSTMModel):
    """A model that uses Attentive Pooling with LSTM as described by Dos Santos et al. (2016)."""
    def build(self, data, sess):
        self.build_input(data, sess)
        self.initialize_weights()

        raw_representation_question = self.bilstm_representation_raw(
            self.embeddings_question,
            self.input_question,
            re_use_lstm=False
        )
        raw_representation_answer_good = self.bilstm_representation_raw(
            self.embeddings_answer_good,
            self.input_answer_good,
            re_use_lstm=True
        )
        raw_representation_answer_bad = self.bilstm_representation_raw(
            self.embeddings_answer_bad,
            self.input_answer_bad,
            re_use_lstm=True
        )

        self.question_good_pooling_weight, self.answer_good_importance_weight = attentive_pooling_weights(
            self.U_AP,
            raw_representation_question,
            raw_representation_answer_good,
            self.input_question,
            self.input_answer_good
        )
        self.question_bad_pooling_weight, self.answer_bad_importance_weight = attentive_pooling_weights(
            self.U_AP,
            raw_representation_question,
            raw_representation_answer_bad,
            self.input_question,
            self.input_answer_bad
        )

        # The question pooling weight has to be set to the question<->good answer weights because this will be the
        # comparison that is relevant for the prediction which will be shown in the web-frontend
        self.question_importance_weight = self.question_good_pooling_weight

        pooled_representation_question_good = weighted_pooling(
            raw_representation_question, self.question_good_pooling_weight, self.input_question
        )
        pooled_representation_answer_good = weighted_pooling(
            raw_representation_answer_good, self.answer_good_importance_weight, self.input_answer_good
        )
        pooled_representation_question_bad = weighted_pooling(
            raw_representation_question, self.question_bad_pooling_weight, self.input_question
        )
        pooled_representation_answer_bad = weighted_pooling(
            raw_representation_answer_bad, self.answer_bad_importance_weight, self.input_answer_bad
        )

        self.create_outputs(
            pooled_representation_question_good,
            pooled_representation_answer_good,
            pooled_representation_question_bad,
            pooled_representation_answer_bad
        )

    def initialize_weights(self):
        """Global initialization of weights for the representation layer

        """
        super(AttentivePoolingLSTMModel, self).initialize_weights()
        self.U_AP = weight_variable('U_AP', [self.lstm_cell_size * 2, self.lstm_cell_size * 2])


component = AttentivePoolingLSTMModel
