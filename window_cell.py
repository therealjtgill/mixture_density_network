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

  def __init__(self, input_size, num_windows=3):
    '''
    Initialize things.
    '''
    print("windowcell num_windows:", num_windows)
    self.NUM_FREE_PARAMS = 3 # Do not change :(
    self._input_size = input_size
    self._state_size = num_windows
    self._output_size = self.NUM_FREE_PARAMS*num_windows
    self.num_windows = num_windows
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
    The output size contains 'alpha', 'beta', and 'kappa' values for each
    component in the window cell.
    '''

    return self._output_size

  def __call__(self, inputs, state, scope=None):
    '''
    Inputs have shape [batch_size, output_size]
    State has shape [batch_size, state_size]
    Splitting of the input into meaningful parameters has to be performed
    here. Since there's no easy way to bring one-hot encodings of characters
    into the RNNCell *and* align them with batch numbers, this cell simply
    returns the RNN-ified parameters of the window cell.
    '''

    with vs.variable_scope(scope or 'window_cell'):
      resized_output = tf.matmul(inputs, self.weight) + self.bias
      print("windowcell inputs info:", inputs)
      [alphas, betas, kappas] = array_ops.split(resized_output, [self._state_size,]*self.NUM_FREE_PARAMS, axis=1)
      kappa_hats = gen_math_ops.exp(kappas - 5) + state
      alpha_hats = gen_math_ops.exp(alphas - 5)
      #alpha_hats = nn_ops.softmax(alphas, axis=1)
      beta_hats = gen_math_ops.exp(betas - 5)
      output = array_ops.concat([alpha_hats, beta_hats, kappa_hats], axis=1)

      return output, kappa_hats

if __name__ == "__main__":

  print("This script inherits the RNNCell interface to create an attention-mechanism-ish RNN.")
  print("Nothing to run this script directly :'( ")