import importlib
import logging
import sys

import click
import numpy as np
import tensorflow as tf

from experiment.config import load_config


@click.command()
@click.argument('config_file')
def run(config_file):
    """This program is the starting point for every experiment. It pulls together the configuration and all necessary
    experiment classes to load

    """
    config = load_config(config_file)
    config_global = config['global']

    # setup a logger
    logger = logging.getLogger('experiment')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler_stdout = logging.StreamHandler(sys.stdout)
    handler_stdout.setLevel(config['logger']['level'])
    handler_stdout.setFormatter(formatter)
    logger.addHandler(handler_stdout)

    if 'path' in config['logger']:
        handler_file = logging.FileHandler(config['logger']['path'])
        handler_file.setLevel(config['logger']['level'])
        handler_file.setFormatter(formatter)
        logger.addHandler(handler_file)

    logger.setLevel(config['logger']['level'])

    # Allow the gpu to be used in parallel
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    if 'max_threads' in config_global:
        sess_config.intra_op_parallelism_threads = config_global['max_threads']

    # we allow to set the random seed in the config file for reproducibility. However, when running on GPU, results
    # will still be nondeterministic (due to nondeterministic behavior of tensorflow)
    if 'random_seed' in config_global:
        seed = config_global['random_seed']
        logger.info('Using fixed random seed'.format(seed))
        np.random.seed(seed)
        tf.set_random_seed(seed)

    with tf.Session(config=sess_config) as sess:
        # We are now fetching all relevant modules. It is strictly required that these module contain a variable named
        # 'component' that points to a class which inherits from experiment.Data, experiment.Experiment,
        # experiment.Trainer or experiment.Evaluator
        data_module = config['data-module']
        model_module = config['model-module']
        training_module = config['training-module']
        evaluation_module = config.get('evaluation-module', None)

        # The modules are now dynamically loaded
        DataClass = importlib.import_module(data_module).component
        ModelClass = importlib.import_module(model_module).component
        TrainingClass = importlib.import_module(training_module).component
        EvaluationClass = importlib.import_module(evaluation_module).component if evaluation_module else None

        # We then wire together all the modules and start training
        data = DataClass(config['data'], config_global, logger)
        model = ModelClass(config['model'], config_global, logger)
        training = TrainingClass(config['training'], config_global, logger)

        # setup the data (validate, create generators, load data, or else)
        logger.info('Setting up the data')
        data.setup()
        # build the model (e.g. compile it)
        logger.info('Building the model')
        model.build(data, sess)
        # start the training process
        logger.info('Starting the training process')
        training.start(model, data, sess)

        # perform evaluation, if required
        if EvaluationClass:
            logger.info('Evaluating')
            evaluation = EvaluationClass(config['evaluation'], config_global, logger)
            evaluation.start(model, data, sess)
        else:
            logger.info('No evaluation')

        logger.info('DONE')


if __name__ == '__main__':
    run()
