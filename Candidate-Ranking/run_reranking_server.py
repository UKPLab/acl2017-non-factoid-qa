import importlib
import logging
import math
import sys

import click
import numpy as np
import tensorflow as tf
from flask import Flask
from flask import json
from flask import request
from nltk import word_tokenize

from experiment.config import load_config
from experiment.qa.data.models import Sentence
from experiment.qa.data.models import Token, TextItem

# we create a flask app for our microservice
app = Flask(__name__)
sess = None
data = None
config = None
model = None


@app.route('/rank', methods=['POST'])
def rerank():
    n = int(request.args.get('n', 10))  # number of answers to return
    input_data = request.get_json()
    question = input_data['question']  # the input question
    candidates = input_data['candidates']  # the input candidate answers

    question_item = get_text_item(question)  # tokenized question
    candidate_items = [get_text_item(c) for c in candidates]  # tokenized candidate answers

    question_vectors = np.array(
        [data.get_item_vector(question_item, config['global']['question_length'])] * len(candidate_items)
    )
    candidate_vectors = np.array(
        [data.get_item_vector(a, config['global']['answer_length']) for a in candidate_items]
    )

    scores = []
    q_weights = []
    a_weights = []
    batch_size = 128
    for batch in range(int(math.ceil(len(candidate_items) / float(batch_size)))):
        batch_indices = batch_size * batch, batch_size * (batch + 1)
        batch_scores, batch_q_weights, batch_a_weights = sess.run(
            [model.predict, model.question_importance_weight, model.answer_good_importance_weight],
            feed_dict={
                model.input_question: question_vectors[batch_indices[0]:batch_indices[1]],
                model.input_answer_good: candidate_vectors[batch_indices[0]:batch_indices[1]],
                model.dropout_keep_prob: 1.0,
            })
        scores += batch_scores.tolist()
        q_weights += batch_q_weights.tolist()
        a_weights += batch_a_weights.tolist()

    # a list of candidate answer indices, sorted by score
    sort_indices = [i for (s, i) in
                    sorted(zip(scores, [i for (i, _) in enumerate(candidates)]), key=lambda x: -x[0])[:n]]

    # the result is a simple json object with the desired content
    result = {
        'question': {
            'tokens': [t.text for t in question_item.sentences[0].tokens]
        },
        'answers': [
            {
                'tokens': [t.text for t in candidate_items[i].sentences[0].tokens],
                'weights': a_weights[i],
                'questionWeights': q_weights[i]
            }
            for i in sort_indices]
    }
    return json.dumps(result)


@app.route('/individual-weights', methods=['POST'])
def individual_weights():
    input_data = request.get_json()
    question = input_data['question']  # the input question
    candidate = input_data['candidate']  # the input candidate answer

    question_item = get_text_item(question)  # tokenized question
    candidate_item = get_text_item(candidate)  # tokenized candidate

    question_vectors = np.array(
        [data.get_item_vector(question_item, config['global']['question_length'])]
    )
    candidate_vectors = np.array(
        [data.get_item_vector(candidate_item, config['global']['answer_length'])]
    )

    q_weights, c_weights = sess.run(
        [model.question_importance_weight, model.answer_good_importance_weight],
        feed_dict={
            model.input_question: question_vectors,
            model.input_answer_good: candidate_vectors,
            model.dropout_keep_prob: 1.0,
        })

    result = {
        'question': {
            'tokens': [t.text for t in question_item.sentences[0].tokens]
        },
        'candidate': {
            'tokens': [t.text for t in candidate_item.sentences[0].tokens],
            'weights': c_weights[0].tolist(),
            'questionWeights': q_weights[0].tolist()
        }
    }
    return json.dumps(result)


def get_text_item(text):
    """Converts a text into a tokenized text item

    :param text:
    :return:
    """
    if config['data']['lowercased']:
        text = text.lower()
    question_tokens = [Token(t) for t in word_tokenize(text)]
    question_sentence = Sentence(' '.join([t.text for t in question_tokens]), question_tokens)
    return TextItem(question_sentence.text, [question_sentence])


#
# The following functions setup the tensorflow graph and perform the training process, if required.
# Furthermore, the webserver is started up
#

@click.command()
@click.argument('config_file')
@click.option('--port', type=int, default=5001,
              help='the port on which the candidate ranking webserver will listen for connections')
def run(config_file, port):
    """This program is the starting point for every experiment. It pulls together the configuration and all necessary
    experiment classes to load

    """
    global sess, model, config, data

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

    sess = tf.InteractiveSession(config=sess_config)

    # We are now fetching all relevant modules. It is strictly required that these module contain a variable named
    # 'component' that points to a class which inherits from experiment.Data, experiment.Experiment,
    # experiment.Trainer or experiment.Evaluator
    data_module = config['data-module']
    model_module = config['model-module']
    training_module = config['training-module']

    # The modules are now dynamically loaded
    DataClass = importlib.import_module(data_module).component
    ModelClass = importlib.import_module(model_module).component
    TrainingClass = importlib.import_module(training_module).component

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

    app.run(debug=False, port=port, host="0.0.0.0")
    sess.close()


if __name__ == '__main__':
    run()
