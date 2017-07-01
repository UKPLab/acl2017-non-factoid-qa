from __future__ import division

import sys

import cherrypy
import requests
import yaml
from flask import Flask
from flask import json
from flask import render_template
from flask import request
from flask.ext.basicauth import BasicAuth
from requests import Timeout

app = Flask(__name__)
configuration = dict()


@app.route('/')
def index():
    return render_template('ask.html', re_rankers=json.dumps(configuration.get('re_rankers')))


@app.route('/get-answers')
def get_answers():
    service_timeout = configuration.get('service_timeout', 30)
    question = request.args.get('q')
    re_ranker = request.args.get('re_ranker')

    try:
        candidate_retrieval_url = '{}/query'.format(
            configuration.get('candidate_retrieval_url', 'http://localhost:9000')
        )
        candidates = requests.get(candidate_retrieval_url, timeout=service_timeout, params={
            'q': question,
            'n': configuration.get('n_candidates', 500)
        }).json()
    except Timeout as e:
        return 'Candidate retrieval timed out', 500
    except BaseException as e:
        return 'Could not reach the candidate retrieval component', 500

    re_rank_data = {
        'question': question,
        'candidates': candidates['candidates']
    }
    try:
        re_rank_action_url = '{}/rank'.format(_get_re_ranker_url(re_ranker))
        re_rank_result = requests.post(
            re_rank_action_url,
            data=json.dumps(re_rank_data),
            headers={'Content-Type': 'application/json'},
            params={
                'n': configuration.get('n_results', 10)
            },
            timeout=service_timeout
        ).json()
    except TimeoutError as e:
        return 'Re-ranking timed out', 500
    except BaseException as e:
        return 'Could not reach the re-ranker', 500

    return json.dumps(re_rank_result)


@app.route('/weights', methods=['POST'])
def get_weights():
    service_timeout = configuration.get('service_timeout', 30)
    re_ranker = request.args.get('re_ranker')
    question = request.json['question']
    candidate = request.json['candidate']

    text_data = {
        'question': question,
        'candidate': candidate
    }
    try:
        action_url = '{}/individual-weights'.format(_get_re_ranker_url(re_ranker))
        weights_result = requests.post(
            action_url,
            data=json.dumps(text_data),
            headers={'Content-Type': 'application/json'},
            timeout=service_timeout
        ).json()
    except TimeoutError as e:
        return 'Re-ranking timed out', 500
    except BaseException as e:
        return 'Could not reach the re-ranker', 500

    return json.dumps(weights_result)


def _get_re_ranker_url(label):
    return [r['url'] for r in configuration.get('re_rankers') if r['label'] == label][0]


if __name__ == '__main__':
    with open(sys.argv[1] if len(sys.argv) > 1 else 'config.yaml', 'r') as f:
        configuration = yaml.load(f)

    debug = configuration.get('debug', False)
    port = configuration.get('port', 5000)
    host = configuration.get('host', '0.0.0.0')

    authentication_configuration = configuration.get('authentication', {})

    app.config['BASIC_AUTH_USERNAME'] = authentication_configuration.get('username', 'username')
    app.config['BASIC_AUTH_PASSWORD'] = authentication_configuration.get('password', 'password')
    app.config['BASIC_AUTH_FORCE'] = authentication_configuration.get('force_all', False)

    basic_auth = BasicAuth(app)

    cherrypy.tree.graft(app, '/')

    # Set the configuration of the web server
    cherrypy.config.update({
        'engine.autoreload_on': debug,
        'server.socket_port': port,
        'server.socket_host': host
    })

    # Start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()
