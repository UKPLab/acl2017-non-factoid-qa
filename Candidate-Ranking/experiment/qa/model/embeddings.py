import tensorflow as tf

from experiment.qa.model import QAModel


class EmbeddingsModel(QAModel):
    """A simple model that learns a representation by averaging over trained word embeddings"""

    def build(self, data, sess):
        self.build_input(data, sess)

        pooled_question = tf.reduce_mean(self.embeddings_question, [1], keep_dims=False)
        pooled_answer_good = tf.reduce_mean(self.embeddings_answer_good, [1], keep_dims=False)
        pooled_answer_bad = tf.reduce_mean(self.embeddings_answer_bad, [1], keep_dims=False)

        self.create_outputs(
            pooled_question,
            pooled_answer_good,
            pooled_question,
            pooled_answer_bad
        )


component = EmbeddingsModel
