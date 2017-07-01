class ComponentBase(object):
    def __init__(self, config, config_global, logger):
        """This is a simple base object for all experiment components

        :type config: dict
        :type config_global: dict
        :type logger: logging.Logger
        """
        self.config = config or dict()
        self.config_global = config_global or dict()
        self.logger = logger


class Data(ComponentBase):
    def setup(self):
        pass

    def get_fold_data(self, fold_i, n_folds):
        """Generates and returns a new Data instance that contains only the data for a specific fold. This method is
        used for hyperparameter optimization on multiple folds.

        :param fold_i: the number of the current fold
        :param n_folds: the total number of folds
        :return: the data for the specified fold
        """
        raise NotImplementedError()


class Model(ComponentBase):
    def __init__(self, config, config_global, logger):
        super(Model, self).__init__(config, config_global, logger)
        self.__summary = None

    def build(self, data, sess):
        raise NotImplementedError()


class Training(ComponentBase):
    def start(self, model, data, sess):
        """

        :param model:
        :type model: Model
        :param data:
        :type data: Data
        """
        raise NotImplementedError()

    def remove_checkpoints(self):
        """Removes all the persisted checkpoint data that was generated during training for restoring purposes"""
        raise NotImplementedError()


class Evaluation(ComponentBase):
    def start(self, model, data, sess):
        """

        :type model: Model
        :type data: Data
        :type sess: tensorflow.Session
        :return:
        """
        raise NotImplementedError()


class EvaluationProductOfExperts(ComponentBase):
    def start(self, models, data):
        """

        :type models: list[(tensorflow.Session, Model)]
        :param data: Data
        :return:
        """
        raise NotImplementedError()
