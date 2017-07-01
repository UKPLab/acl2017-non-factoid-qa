import tensorflow as tf

import experiment


class QANoTraining(experiment.Training):
    """This is a replacement component that skips the training process"""

    def __init__(self, config, config_global, logger):
        super(QANoTraining, self).__init__(config, config_global, logger)

    def start(self, model, data, sess):
        self.logger.info('Initializing all variables')
        sess.run(tf.global_variables_initializer())
        self.logger.info("Skipping training")

    def remove_checkpoints(self):
        pass


component = QANoTraining
