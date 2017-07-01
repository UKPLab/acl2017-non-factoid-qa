import unittest

import numpy as np
import numpy.testing as npt
import tensorflow as tf

from experiment.qa.model.helper.pooling_helper import attention_softmax, non_zero_tokens, soft_alignment, \
    attentive_pooling


class TestPoolingHelper(unittest.TestCase):
    def setUp(self):
        self.sess = tf.InteractiveSession()

    def tearDown(self):
        self.sess.close()

    def test_non_zero_tokens(self):
        tokens = tf.constant([
            [24., 22., 11234., 0., 0.],
            [31., 0., 0., 0., 0.]
        ])
        result = self.sess.run(non_zero_tokens(tokens))
        reference_value = np.array([
            [1., 1., 1., 0., 0.],
            [1., 0., 0., 0., 0.]
        ])

        npt.assert_array_equal(result, reference_value)

    def test_attention_softmax(self):
        vector_in = tf.constant([
            [1., 2., 1., 2.0],
            [.3, .2, .9, .3]
        ])
        padding = tf.constant([
            [1., 1., 1., 0.],
            [1., 1., 0., 0.]
        ])

        result = self.sess.run(attention_softmax(vector_in, padding))
        reference_value = np.array([
            [0.21194156, 0.57611692, 0.21194156, 0.],
            [0.52497919, 0.47502081, 0., 0.]
        ])

        npt.assert_array_almost_equal(result, reference_value)

    def test_soft_alignment(self):
        """Tests the soft alignment function and its capability to handle minibatches with zero-padding"""
        U_AP = tf.constant(
            [
                [1., 1.],
                [1., 1.]
            ]
        )
        raw_question_rep = tf.constant(
            [[
                [.2, .7],
                [.4, .8],
                [.1, .9],
                [.7, .8]
            ]] * 2
        )
        raw_answer_rep = tf.constant(
            [[
                [.3, .9],
                [.5, .9],
                [.7, .6],
                [.9, .7]
            ]] * 2
        )
        tokens_question_non_zero = tf.constant(
            [
                [1., 1., 0., 0.]
            ] * 2
        )
        tokens_answer_non_zero = tf.constant(
            [
                [1., 1., 1., 0.]
            ] * 2
        )

        result = self.sess.run(soft_alignment(
            U_AP, raw_question_rep, raw_answer_rep, tokens_question_non_zero, tokens_answer_non_zero
        ))

        # QU = [[0.9, 0.9], [1.2, 1.2]]
        # QU(A^T) = [[1.08, 1.26, 1.17], [1.44, 1.68, 1.56]]
        # tanh(...) = [[0.7931991, 0.85106411, 0.82427217], [0.89369773, 0.93286155, 0.91542046]]

        # Due to padding, the resulting tensor will have a different shape. We verify that the relevant part of the
        # result has the correct values, and the rest holds values less than -1

        reference_value = np.array(
            [[
                [0.7931991, 0.85106411, 0.82427217],
                [0.89369773, 0.93286155, 0.91542046]
            ]] * 2
        )

        npt.assert_array_almost_equal(result[:, 0:2, 0:3], reference_value)
        npt.assert_array_less(result, np.array(
            [[
                [1.01, 1.01, 1.01, -1.],
                [1.01, 1.01, 1.01, -1.],
                [-1., -1., -1., -1.],
                [-1., -1., -1., -1.]
            ]] * 2
        ))

    def test_attentive_pooling(self):
        """Test the full functionality with the same values as before"""
        U_AP = tf.constant(
            [
                [1., 1.],
                [1., 1.]
            ]
        )
        raw_question_rep = tf.constant(
            [[
                [.2, .7],
                [.4, .8],
                [.1, .9],
                [.7, .8]
            ]] * 2
        )
        raw_answer_rep = tf.constant(
            [[
                [.3, .9],
                [.5, .9],
                [.7, .6],
                [.9, .7]
            ]] * 2
        )
        tokens_question = tf.constant(
            [
                [123, 6, 0., 0.]
            ] * 2
        )
        tokens_answer = tf.constant(
            [
                [33, 1, 12, 0.]
            ] * 2
        )

        result_repr_q, result_repr_a = self.sess.run(attentive_pooling(
            U_AP, 2, raw_question_rep, raw_answer_rep, tokens_question, tokens_answer, 'name'
        ))

        # tanh(...) = [[0.7931991, 0.85106411, 0.82427217], [0.89369773, 0.93286155, 0.91542046]]
        # max over rows = [[0.85106411, 0.93286155]]
        # max over colums = [[0.89369773, 0.93286155, 0.91542046]]

        # attention question = [ 0.47956203,  0.52043797]
        # attention answer = [ 0.32659447,  0.33963892,  0.33376661]

        # question-rep = [0.304088, 0.752044]
        # answer-rep = [0.501434, 0.79987]

        reference_value_repr_q = np.array(
            [
                [0.304088, 0.752044]
            ] * 2
        )
        reference_value_repr_a = np.array(
            [
                [0.501434, 0.79987]
            ] * 2
        )

        npt.assert_array_almost_equal(result_repr_q, reference_value_repr_q)
        npt.assert_array_almost_equal(result_repr_a, reference_value_repr_a)
