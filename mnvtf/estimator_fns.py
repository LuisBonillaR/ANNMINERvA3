from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging
import numpy as np
import tensorflow as tf
from absl import logging

from tensorflow.python.ops import math_ops
from mnvtf.model_classes import ConvModel
from mnvtf.resnet_model import Model
from mnvtf.bilinearloss import bilinear_loss
from tensorflow_estimator.python.estimator.canned import metric_keys

#LOGGER = logging.getLogger(__name__)

# Global variables
conf_mat=np.array(
#    [[0.63231192, 0.30853377, 0.04294344, 0.01109281, 0.00144975, 0.00366831],
#     [0.05506895, 0.70534465, 0.19972489, 0.02841036, 0.00410313, 0.00734801],
#     [0.01005448, 0.23972295, 0.59707759, 0.11422574, 0.01800985, 0.0209094 ],
#     [0.00550186, 0.10522105, 0.34555606, 0.35203748, 0.09558872, 0.09609482],
#     [0.00306038, 0.06638446, 0.14972971, 0.3213683 , 0.20673283, 0.25272431],
#     [0.00209378, 0.04615158, 0.06184017, 0.1481274 , 0.12353288, 0.6182542 ]]

# ANN 50mev
    [[0.65538063, 0.29299783, 0.02893968, 0.0151066 , 0.00171267, 0.00586259],
     [0.05635235, 0.75191917, 0.13698083, 0.03931792, 0.00459659, 0.01083315],
     [0.0107037 , 0.27483446, 0.5190601 , 0.14408765, 0.02094202, 0.03037206],
     [0.00512452, 0.09545647, 0.24814766, 0.40966805, 0.10898587, 0.13261742],
     [0.0028    , 0.05342857, 0.076     , 0.29994286, 0.23897143, 0.32885714],
     [0.00174134, 0.03364618, 0.02789092, 0.09837082, 0.10450977, 0.73384098]]

#    [0.59569718, 0.33732494, 0.04566673, 0.01359854, 0.00324741, 0.00446519],
#    [0.0591597,  0.6521325,  0.2322468,  0.03931633, 0.00698487, 0.01015981],
#    [0.01198968, 0.25868872, 0.57391106, 0.10760358, 0.0199575,  0.02784945],
#    [0.00598415, 0.09833414, 0.38573508, 0.29322335, 0.08054343, 0.13617985],
#    [0.00258993, 0.05841727, 0.1628777,  0.32719424, 0.15338129, 0.29553957],
#    [0.00179695, 0.04073076, 0.0557053,  0.15483678, 0.10212639, 0.64480383]]
)
    
# Get rid of the diagonal part of the pen_mat
conf_mat -= np.eye(conf_mat.shape[0]) * np.diag(conf_mat)
#conf_mat /= conf_mat.sum(axis=1) #1 column "normalize"
conf_mat /= conf_mat.sum(axis=1)[:,np.newaxis] #2 row normalize
#conf_mat /= conf_mat.max() #3
# Need a tf.constant version of the pen_mat
#pen_mat = tf.constant(conf_mat)

def _get_block_sizes(resnet_size):
  """Retrieve the size of each block_layer in the ResNet model.

  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.

  Args:
    resnet_size: The number of convolutional layers needed in the model.

  Returns:
    A list of block sizes to use in building the model.

  Raises:
    KeyError: if invalid resnet_size is received.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)

class EstimatorFns:
    def __init__(self, nclasses=6, cnn_model='ANN'):
        self._nclasses = nclasses
        self._model = cnn_model

    def est_model_fn(self, features, labels, mode, params):
        # Choose model for training
        if self._model == 'ResNetX':
            resnet_size = 50
            model = Model(
                resnet_size=resnet_size,
                bottleneck=True,
                num_classes=self._nclasses,
                num_filters=64,
                kernel_size=3, #7,
                conv_stride=2,
                first_pool_size=3,
                first_pool_stride=2,
                block_sizes=_get_block_sizes(resnet_size),
                block_strides=[1, 2, 2, 2],
#                resnet_version=resnet_version,
                data_format='channels_first'
#                dtype=dtype
                )
            logits = model(features['concat'], training=True)
        
        elif self._model == 'ANN':
            model = ConvModel(self._nclasses)
            logits = model(features['x_img'],
                           features['u_img'],
                           features['v_img'])
    
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'classes': tf.argmax(logits, axis=1),
                'probabilities': tf.nn.softmax(logits),
                'eventids': features['eventids']
            }
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.PREDICT,
                predictions=predictions,
                export_outputs={
                    'classify': tf.estimator.export.PredictOutput(predictions)
                }
            )
        else:
          loss = bilinear_loss(labels, logits, conf_mat, alpha=.50)
#          loss = tf.compat.v1.losses.softmax_cross_entropy(
#              onehot_labels=labels, logits=logits)
          accuracy = tf.compat.v1.metrics.accuracy(
              labels=tf.argmax(labels, axis=1),
              predictions=tf.argmax(logits, axis=1)
          )
    
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.compat.v1.summary.scalar('accuracy', accuracy[1])
            # If we are running multi-GPU, we need to wrap the optimizer!
            if self._model == 'ANN':
                optimizer = tf.compat.v1.train.MomentumOptimizer(
                    learning_rate=0.0025, momentum=0.9, use_nesterov=True
                ) #GradientDescentOptimizer(
            elif self._model == 'ResNetX':
                optimizer = tf.compat.v1.train.AdamOptimizer() #epsilon=1e-08
    
            # Name tensors to be logged with LoggingTensorHook (??)
            tf.identity(loss, 'cross_entropy_loss')
            # Save accuracy scalar to Tensorboard output (loss auto-logged)
            #tf.compat.v1.summary.scalar('train_accuracy', accuracy)
            logging_hook = tf.estimator.LoggingTensorHook(
                {"loss" : loss, "accuracy" : accuracy[1]},
                every_n_iter=500
            )
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.TRAIN,
                loss=loss,
                train_op=optimizer.minimize(
                    loss,  tf.compat.v1.train.get_or_create_global_step()
                ),
                training_hooks = [logging_hook]
            )
    
        if mode == tf.estimator.ModeKeys.EVAL:
            # we get loss 'for free' as an eval_metric
            save_acc_hook = tf.train.SummarySaverHook(
                save_steps=1500,
                output_dir='/data/minerva/JLBRtesthad/tensorflow/models/tests',
                summary_op=tf.compat.v1.summary.scalar('eval_acc', accuracy[1])
            )
            save_loss_hook = tf.train.SummarySaverHook(
                save_steps=1500,
                output_dir='/data/minerva/JLBRtesthad/tensorflow/models/tests',
                summary_op=tf.compat.v1.summary.scalar('eval_loss', loss)
            )
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=loss,
                eval_metric_ops={'accuracy':accuracy},
                evaluation_hooks=[save_acc_hook, save_loss_hook]
            )
    
        return None


def serving_input_receiver_fn():
    """
    This is used to define inputs to serve the model.
    :return: ServingInputReciever
    """
    reciever_tensors = {
        # The size of input image is flexible.
        'eventids' : tf.placeholder(tf.int64, [None]),
        'x_img': tf.placeholder(tf.float32, [None, 2, 127, 94]),
        'v_img': tf.placeholder(tf.float32, [None, 2, 127, 47]),
        'u_img': tf.placeholder(tf.float32, [None, 2, 127, 47]),
    }

    return tf.estimator.export.ServingInputReceiver(
        receiver_tensors=reciever_tensors,
        features=reciever_tensors
        )


def loss_smaller(best_eval_result, current_eval_result):
    """Compares two evaluation results and returns true if the 2nd one is
    smaller.
    Both evaluation results should have the values for MetricKeys.LOSS, which
    are used for comparison.
    Args:
        best_eval_result: best eval metrics.
        current_eval_result: current eval metrics.
    Returns:
        True if the loss of current_eval_result is smaller; otherwise, False.
    Raises:
        ValueError: If input eval result is None or no loss is available.
    """
    default_key = metric_keys.MetricKeys.LOSS
    if not best_eval_result or default_key not in best_eval_result:
        raise ValueError(
            'best_eval_result cannot be empty or no loss is found in it.')
    
    if not current_eval_result or default_key not in current_eval_result:
        raise ValueError(
            'current_eval_result cannot be empty or no loss is found in it.')
    
    return best_eval_result[default_key] > current_eval_result[default_key]
    
