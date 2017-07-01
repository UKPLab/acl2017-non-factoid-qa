import tensorflow as tf


def weighted_pooling(raw_representation, positional_weighting, tokens):
    """Performs a pooling operation that is similar to average pooling, but uses a weight for every position. Therefore
     not all positions are equally important and contribute equally to the resulting vector.

    :param raw_representation:
    :param positional_weighting:
    :param tokens:
    :return:
    """
    positional_weighting_non_zero = non_zero_tokens(tf.to_float(tokens)) * positional_weighting
    pooled_representation = tf.matmul(
        tf.reshape(positional_weighting_non_zero, [-1, 1, tf.shape(positional_weighting)[1]]),
        raw_representation
    )
    return tf.reshape(pooled_representation, [-1, tf.shape(raw_representation)[2]])


def non_zero_tokens(tokens):
    """Receives a vector of tokens (float) which are zero-padded. Returns a vector of the same size, which has the value
    1.0 in positions with actual tokens and 0.0 in positions with zero-padding.

    :param tokens:
    :return:
    """
    return tf.ceil(tokens / tf.reduce_max(tokens, [1], keep_dims=True))


def soft_alignment(U_AP, raw_question_rep, raw_answer_rep, tokens_question_non_zero, tokens_answer_non_zero):
    """Calculate the AP soft-alignment matrix (in a batch-friendly fashion)

    :param U_AP: The AP similarity matrix (to be learned)
    :param raw_question_rep:
    :param raw_answer_rep:
    :param tokens_question_non_zero:
    :param tokens_answer_non_zero:
    :return:
    """
    answer_transposed = tf.transpose(raw_answer_rep, [0, 2, 1])

    # Unfortunately, there is no clean way in TF to multiply a 3d tensor with a 2d tensor. We need to perform some
    # reshaping. Compare solution 2 on
    # http://stackoverflow.com/questions/38235555/tensorflow-matmul-of-input-matrix-with-batch-data
    raw_question_rep_flat = tf.reshape(raw_question_rep, [-1, tf.shape(raw_question_rep)[2]])
    QU_flat = tf.matmul(raw_question_rep_flat, U_AP)
    QU = tf.reshape(QU_flat, [-1, tf.shape(raw_question_rep)[1], tf.shape(raw_question_rep)[2]])
    QUA = tf.matmul(QU, answer_transposed)
    G = tf.nn.tanh(QUA)

    # We are now removing all the fields of G that belong to zero padding. To achieve this, we are determining these
    # fields and adding a value of -2 to all of them (which is guaranteed to result in a smaller number than the minimum
    # of G, which is -1)
    additions_G_question = tf.transpose(
        tf.reshape((tokens_question_non_zero - 1) * 2, [-1, 1, tf.shape(tokens_question_non_zero)[1]]),
        [0, 2, 1]
    )
    additions_G_answer = tf.reshape((tokens_answer_non_zero - 1) * 2, [-1, 1, tf.shape(tokens_answer_non_zero)[1]])

    # G_non_zero contains values of less than -1 for all fields which have a relation to zero-padded token positions
    G_non_zero = G + additions_G_question + additions_G_answer

    return G_non_zero


def attention_softmax(attention, indices_non_zero):
    """Our very own softmax that ignores values of zero token indices (zero-padding)

    :param attention:
    :param raw_indices:
    :return:
    """
    ex = tf.exp(attention) * indices_non_zero
    sum = tf.reduce_sum(ex, [1], keep_dims=True)
    softmax = ex / sum
    return softmax


def maxpool_tanh(item, tokens):
    """Calculates the max-over-time, but ignores the zero-padded positions

    :param item:
    :param tokens:
    :return:
    """
    non_zero = non_zero_tokens(tf.to_float(tokens))
    additions = tf.reshape(
        (non_zero - 1) * 2,
        [-1, tf.shape(tokens)[1], 1]
    )

    # additions push the result below -1, which can not be selected in maxpooling
    tanh_item = tf.nn.tanh(item)
    item_processed = tanh_item + additions

    return maxpool(item_processed)


def maxpool(item):
    return tf.reduce_max(item, [1], keep_dims=False)


def attentive_pooling_weights(U_AP, raw_question_rep, raw_answer_rep, tokens_question, tokens_answer,
                              apply_softmax=True):
    """Calculates the attentive pooling weights for question and answer

    :param U_AP: the soft-attention similarity matrix (to learn)
    :param raw_question_rep:
    :param raw_answer_rep:
    :param tokens_question: The raw token indices of the question. Used to detection not-set tokens
    :param tokens_answer: The raw token indices of the answer. Used to detection not-set tokens
    :param Q_PW: Positional weighting matrix for the question
    :param A_PW: Positional weighting matrix for the answer
    :param apply_softmax:
    :return: question weights, answer weights
    """
    tokens_question_float = tf.to_float(tokens_question)
    tokens_answer_float = tf.to_float(tokens_answer)
    tokens_question_non_zero = non_zero_tokens(tokens_question_float)
    tokens_answer_non_zero = non_zero_tokens(tokens_answer_float)

    G = soft_alignment(U_AP, raw_question_rep, raw_answer_rep, tokens_question_non_zero, tokens_answer_non_zero)

    maxpool_GQ = tf.reduce_max(G, [2], keep_dims=False)
    maxpool_GA = tf.reduce_max(G, [1], keep_dims=False)

    if apply_softmax:
        attention_Q = attention_softmax(maxpool_GQ, tokens_question_non_zero)
        attention_A = attention_softmax(maxpool_GA, tokens_answer_non_zero)
    else:
        attention_Q = maxpool_GQ
        attention_A = maxpool_GA

    return attention_Q, attention_A
