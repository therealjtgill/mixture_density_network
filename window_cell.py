from __future__ import print_function

import numpy as np
from numpy.random import rand

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.ops import random_ops
import tensorflow as tf

from tensorflow.python.ops.rnn_cell_impl import RNNCell as RNNCell

class WindowCell(RNNCell):

  def __init__(self, input_size, num_chars, num_windows=3):
    '''
    Initialize things.
    '''
    print("windowcell num_windows:", num_windows)
    self.NUM_FREE_PARAMS = 3 # Do not change :(
    self._input_size = input_size
    self._state_size = num_windows
    self._output_size = self.NUM_FREE_PARAMS*num_windows
    self.num_windows = num_windows
    self.num_chars = num_chars
    self.weight = variables.Variable(random_ops.random_normal(shape=(self._input_size, self._output_size), stddev=0.1), dtype=tf.float32)
    self.bias = variables.Variable(random_ops.random_normal(shape=(self._output_size,), stddev=0.1), dtype=tf.float32)

  @property
  def state_size(self):
    '''
    Recurrent state only contains 'kappa' values from the window mixture.
    The 'alpha' and 'beta' values are passed in as inputs, but are ignored
    in the recurrence relation.
    '''

    return self._state_size

  @property
  def output_size(self):
    '''
    The output has the same size as phi at the given timestep, which means that
    phi is self.num_chars long.
    '''

    return self.num_chars

  def __call__(self, inputs, state, scope=None):
    '''
    Inputs have shape [batch_size, output_size]
    State has shape [batch_size, state_size]
    Splitting of the input into meaningful parameters has to be performed
    here. Since there's no easy way to bring one-hot encodings of characters
    into the RNNCell *and* align them with batch numbers, this cell simply
    returns the RNN-ified parameters of the window cell.
    '''

    dtype = tf.float32

    with vs.variable_scope(scope or 'window_cell'):
      resized_input = tf.matmul(inputs, self.weight) + self.bias
      #print("windowcell inputs info:", inputs)
      [alphas, betas, kappas] = array_ops.split(resized_input, [self._state_size,]*self.NUM_FREE_PARAMS, axis=1)
      kappa_hats = gen_math_ops.exp(kappas) + state
      alpha_hats = gen_math_ops.exp(alphas)
      beta_hats = gen_math_ops.exp(betas)
      #beta_hats = 8*gen_math_ops.sigmoid(betas) + 0.1
      u = tf.range(tf.cast(self.num_chars, dtype), dtype=dtype) # Integer values of 'u' in phi

      kappa_hat_list = tf.split(kappa_hats, [1,]*self.num_windows, axis=1)
      beta_hat_list = tf.split(beta_hats, [1,]*self.num_windows, axis=1)
      alpha_hat_list = tf.split(alpha_hats, [1,]*self.num_windows, axis=1)

      phi = 0
      for i in range(self.num_windows):
        kappa_hat_tiled = tf.tile(kappa_hat_list[i], [1, self.num_chars])
        beta_hat_tiled = tf.tile(beta_hat_list[i], [1, self.num_chars])
        alpha_hat_tiled = tf.tile(alpha_hat_list[i], [1, self.num_chars])
        z = -1*beta_hat_tiled*tf.square(kappa_hat_tiled - u)
        phi += alpha_hat_tiled*tf.exp(z)
      print("information about phi:", phi)
      return phi, kappa_hats

if __name__ == "__main__":

  print("This script inherits the RNNCell interface to create an attention-mechanism-ish RNN.")
  print("Nothing to run this script directly :'( ")