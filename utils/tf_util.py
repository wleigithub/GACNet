""" Wrapper functions for TensorFlow layers.
"""

import numpy as np
import tensorflow as tf

def variable_with_weight_decay(name, shape, wd, trainable=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def get_bias(shape, trainable=True):
    return tf.get_variable(name='biases', shape=shape, dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=trainable)

def conv1d(inputs,num_output_channels,kernel_size,scope, stride=1,padding='SAME',weight_decay=0.0,bn=False,
            bn_decay=None,is_training=None, activation_fn=tf.nn.elu, trainable=True):
  """ 1D convolution with non-linear operation.
  Args:
    inputs: 3-D tensor variable BxLxC
    num_output_channels: int
    kernel_size: int
    scope: string
    stride: int
    padding: 'SAME' or 'VALID'
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    num_in_channels = inputs.get_shape()[-1].value
    kernel_shape = [kernel_size, num_in_channels, num_output_channels]
    kernel = variable_with_weight_decay('weights', shape=kernel_shape, wd=weight_decay, trainable=trainable)
    outputs = tf.nn.conv1d(inputs, kernel,stride=stride,padding=padding)
    biases = get_bias([num_output_channels], trainable)
    outputs = tf.nn.bias_add(outputs, biases)

    if bn:
      outputs = batch_norm(outputs, is_training, bn_decay=bn_decay, scope='bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


def conv2d(inputs, num_output_channels,kernel_size,scope,stride=[1, 1],padding='SAME',weight_decay=0.0,bn=False, 
            bn_decay=None,is_training=None, activation_fn=tf.nn.elu, trainable=True):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w, num_in_channels, num_output_channels]
      kernel = variable_with_weight_decay('weights',shape=kernel_shape, wd=weight_decay, trainable=trainable)
      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(inputs, kernel,[1, stride_h, stride_w, 1],padding=padding)
      biases = get_bias([num_output_channels], trainable)
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = batch_norm(outputs, is_training, bn_decay=bn_decay, scope='bn')

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs


def MLP_2d(features, mlp, is_training, bn_decay, bn, scope='mlp2d', padding='VALID'):
    for i, num_out_channel in enumerate(mlp):
        features = conv2d(features, num_out_channel, [1,1],
                                            padding=padding, stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope= scope+'_%d'%(i), bn_decay=bn_decay) 
    return features


def MLP_1d(features, mlp, is_training, bn_decay, bn, scope='mlp1d', padding='VALID'):
    for i, num_out_channel in enumerate(mlp):
        features = conv1d(features, num_out_channel, 1,
                                            padding=padding, stride=1,
                                            bn=bn, is_training=is_training,
                                            scope=scope+'_%d'%(i), bn_decay=bn_decay) 
    return features


def fully_connected(inputs,num_outputs,scope,weight_decay=0.0,activation_fn=tf.nn.elu,
                    bn=False, bn_decay=None,is_training=None, trainable=True):
  """ Fully connected layer with non-linear operation.
  Args:
    inputs: 2-D tensor BxN
    num_outputs: int
  
  Returns:
    Variable tensor of size B x num_outputs.
  """
  with tf.variable_scope(scope) as sc:
    num_input_units = inputs.get_shape()[-1].value
    weights = variable_with_weight_decay('weights',shape=[num_input_units, num_outputs],wd=weight_decay, trainable=trainable)
    outputs = tf.matmul(inputs, weights)
    biases = get_bias([num_outputs], trainable)
    outputs = tf.nn.bias_add(outputs, biases)
     
    if bn:
      outputs = batch_norm(outputs, is_training, bn_decay, 'bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
  """ Dropout layer.

  Args:
    inputs: tensor
    is_training: boolean tf.Variable
    scope: string
    keep_prob: float in [0,1]
    noise_shape: list of ints

  Returns:
    tensor variable
  """
  with tf.variable_scope(scope) as sc:
    outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
    return outputs


def batch_norm(data, is_training, bn_decay, scope):
    decay = bn_decay if bn_decay is not None else 0.5
    return tf.layers.batch_normalization(data, momentum=decay, training=is_training,
                                         beta_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                         gamma_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                         name=scope)


def _batch_norm(inputs, is_training, bn_decay, scope):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[inputs.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[inputs.shape[-1]]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, range(len(inputs.shape) - 1), name='moments')

        decay = bn_decay if bn_decay is not None else 0.5
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_training, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed


def get_learning_rate(batch, BASE_LEARNING_RATE, BATCH_SIZE, DECAY_STEP, DECAY_RATE):
    """Learning rate decay.
    """
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate        


def get_bn_decay(batch, BATCH_SIZE, BN_DECAY_DECAY_STEP=300000.0, BN_DECAY_DECAY_RATE=0.5, BN_INIT_DECAY = 0.5, BN_DECAY_CLIP = 0.99):
    """
    BN_DECAY_DECAY_STEP: accept BN_DECAY_DECAY_STEP with float type
    """
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay