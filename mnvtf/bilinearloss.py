import numpy as np
import tensorflow as tf
import logging

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

LOGGER = logging.getLogger(__name__)

def bilinear_loss(
    onehot_labels, logits, pen_mat=None, alpha=.5,
    log=False, scope=None, tf_sfmx_ce=False):
    """Generate Bilinear/Log-Bilinear loss functions combined with the regular
    cross-entorpy loss
    (1 - alpha)*cross_entropy_loss + alpha*bilinar/log-bilinar
    `weights` acts as a coefficient for the loss. If a scalar is provided,
    then the loss is simply scaled by the given value. If `weights` is a
    tensor of shape `[batch_size]`, then the loss weights apply to each
    corresponding sample.
    If `label_smoothing` is nonzero, smooth the labels towards 1/num_classes:
        new_onehot_labels = onehot_labels * (1 - label_smoothing)
                            + label_smoothing / num_classes
    Note that `onehot_labels` and `logits` must have the same shape,
    e.g. `[batch_size, num_classes]`. The shape of `weights` must be
    broadcastable to loss, whose shape is decided by the shape of `logits`.
    In case the shape of `logits` is `[batch_size, num_classes]`, loss is
    a `Tensor` of shape `[batch_size]`.
    Args:
        onehot_labels: One-hot-encoded labels.
        logits: Logits outputs of the network.
        pen_mat: `np.Array`. All positive penalty matrix. A higher value in
            [i, j] indicates a higher penalty for making the mistake of
            classifying an example really of class i, as class j (i.e. placing
            weight there, since the output is a probability vector).
        alpha: float. The trade-off paramter between the cross-entropy and 
            bilinear/log-bilinear parts of the loss.
        log: bool. wether to generate the log-blinear loss
        tf_sfmx_ce: If set to True, we get loss from
            tf.nn.softmax_cross_entropy_with_logits_v2, else we take custom
            softmax cross entropy.
    Raises:
        ValueError: If the shape of `logits` doesn't match that of
            `onehot_labels` or if the shape of `weights` is invalid or if
            `weights` is None.  Also if `onehot_labels`, `logits`, `pen_mat` is
            None, or alpha is not 0 <= alpha <= 1.
    Returns:
        Weighted loss escalar `Tensor` of the same type as `logits`.
    """
    if onehot_labels is None:
        raise ValueError("onehot_labels must not be None.")
    if logits is None:
        raise ValueError("logits must not be None.")
    if pen_mat is None:
        raise ValueError("pen_mat must not be None.")
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must take values between 0 and 1.")
    
    with ops.name_scope(scope, "bilinear_loss",
                      (logits, onehot_labels, pen_mat)) as scope:
        logits = ops.convert_to_tensor(logits)
        onehot_labels = math_ops.cast(onehot_labels, logits.dtype)
        logits.get_shape().assert_is_compatible_with(
            onehot_labels.get_shape())
        pen_mat = math_ops.cast(pen_mat, logits.dtype)

        onehot_labels = array_ops.stop_gradient(
            onehot_labels, name="labels_stop_gradient")
        pen_mat = array_ops.stop_gradient(
            pen_mat, name="pen_mat_stop_gradient")
        alpha = array_ops.stop_gradient(
            alpha, name="alpha_stop_gradient")
        
        # The regular cross-entropy loss
        logits_safe = logits - tf.reduce_max(logits, axis=1)[:, tf.newaxis]
        logsoftmax = logits_safe - tf.math.log(
            tf.math.reduce_sum(
                tf.math.exp(logits_safe), axis=1)[:, tf.newaxis])
        softmax = tf.math.exp(logsoftmax) if alpha != 0 else 0
        if tf_sfmx_ce:
            diagonal_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=onehot_labels, logits=logits, name="xentropy")
        else:
            diagonal_loss = -tf.math.reduce_sum(
                tf.math.multiply(onehot_labels, logsoftmax), axis=1)

        # The off-disgonal part of the loss -- how we weigh the error i->j
        if alpha != 0 and log:
            off_diagonal_loss = -tf.math.reduce_sum(
                tf.math.multiply(
                    tf.linalg.matmul(onehot_labels, pen_mat),
                    tf.math.log(1 - softmax + 1e-10)),
                axis=1)
        elif alpha != 0 and not log:
            off_diagonal_loss = tf.math.reduce_sum(
                tf.math.multiply(
                    tf.linalg.matmul(onehot_labels, pen_mat),
                    softmax),
                axis=1)
        else:
            off_diagonal_loss = 0

        return tf.compat.v1.losses.compute_weighted_loss(
            diagonal_loss*(1-alpha) + off_diagonal_loss*alpha, scope=scope)
