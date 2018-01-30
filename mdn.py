import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class MDN(object):
  '''
  Contains useful methods for training, testing, and validating a mixture
  density network.
  '''

  def __init__(self, session, input_size, num_gaussians=3):
    '''
    Sets up the computation graph for the MDN.
    Bishop, et. al, use a mixture of univariate gaussians, which allows them
    to avoid futzing around with off-diagonal covariance matrix terms. It also
    affords them a bit of extra numerical stability, since they don't have to
    worry about diagonal elements going to zero (exponentiate the node
    containing the single variance term).
    '''

    dtype = tf.float32
    self.session = session
    self.weights = []
    self.biases = []
    self.layers = []
    self.init_states = []
    self.num_lstm_layers = 2
    self.num_gaussians = num_gaussians
    num_lstm_cells = 200
    num_means = num_gaussians*(input_size - 1)
    num_variances = num_gaussians*(input_size - 1)
    output_size = num_gaussians + num_means + num_variances + 1
    print('output size:', output_size, (output_size - 1)/num_gaussians)

    with tf.variable_scope("mdn"):

      self.input_data = tf.placeholder(dtype=dtype,
                                       shape=[None, None, input_size])
      self.output_data = tf.placeholder(dtype=dtype,
                                        shape=[None, None, output_size])

      # For each layer of lstm's, create a set of placeholders to contain
      # values passed to each lstm cell's initial recurrent state.
      for i in range(self.num_lstm_layers):
        ph_c = tf.placeholder(dtype=dtype, shape=[None, num_lstm_cells])
        ph_h = tf.placeholder(dtype=dtype, shape=[None, num_lstm_cells])
        self.init_states.append(
          tf.nn.rnn_cell.LSTMStateTuple(ph_c, ph_h))

      self.init_states = tuple(self.init_states)

      #input_data_flat = tf.reshape(self.input_data, [-1, input_size])

      #shape = (input_size, 200)
      #self.layers.append(tf.nn.relu(self._linear_op(input_data_flat, shape)))

      for i in range(self.num_lstm_layers):
        self.layers.append(tf.nn.rnn_cell.BasicLSTMCell(num_lstm_cells))

      # Get a list of LSTM cells in the current set of layers, then pass those
      # to the MultiRNNCell method.
      lstm_layers = [l for l in self.layers if "BasicLSTMCell" in str(type(l))]
      self.multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layers)

      # LSTM layers
      outputs, self.last_lstm_state = \
        tf.nn.dynamic_rnn(self.multi_lstm_cell, self.input_data, dtype=dtype,
                          initial_state=self.init_states)
      outputs_flat = tf.reshape(outputs, [-1, num_lstm_cells])
      self.layers.append(outputs_flat)

      # Output layer
      shape = [num_lstm_cells, output_size]
      self.layers.append(self._linear_op(self.layers[-1], shape))

      # Get the mixture components
      splits = [num_means, num_variances, num_gaussians, 1]
      pieces = tf.split(self.layers[-1], splits, axis=1)
      self.means = pieces[0]
      self.stdevs = tf.nn.softplus(pieces[1])
      self.mix_weights = tf.nn.softmax(pieces[2])
      self.stroke = tf.nn.sigmoid(pieces[3])

      outputs_flat = tf.reshape(self.output_data, [-1, input_size])

      gauss_values, stroke = tf.split(outputs_flat, [input_size-1, 1], axis=1)

      self.gauss_evals = self._eval_gaussians(gauss_values, self.means,
                                           self.stdevs, num_gaussians)
      self.gauss_params = self._get_gaussian_params(self.means, self.stdevs,
                                                    num_gaussians)
      self.mixture = tf.reduce_sum(self.gauss_evals*self.mix_weights, axis=-1)

      self.loss = -1*tf.reduce_mean(tf.log(self.mixture))
      # Need to clip gradients (?)
      optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
      self.train_op = optimizer.minimize(self.loss)


  def _get_weights(self, shape, name="requested_weight"):
    '''
    Returns a location of a Rank(2) tensor of weights and a Rank(1) tensor of
    biases within the self.weights and self.biases lists.
    '''

    weights = tf.Variable(tf.random_normal(shape, stddev=0.1), name=name)
    biases = tf.Variable(tf.random_normal([shape[-1]], stddev=0.1), name=name)

    self.weights.append(weights)
    self.biases.append(biases)

    return (len(self.weights) - 1, len(self.biases) - 1)


  def _linear_op(self, input_tensor, shape):
    '''
    Perform simple matmul and bias offset between the input_tensor and a tensor
    of weights that will be generated in this method.
    So you specify an input tensor and the shape of the weight matrix, and this
    method does the following:

      create weight: W with shape = "shape"
      create bias: b with shape = "shape[1]"

      matmul(input_tensor, W) + b
    '''

    (W_loc, b_loc) = self._get_weights(shape, "linear_op_weights")

    return tf.matmul(input_tensor, self.weights[W_loc]) + self.biases[b_loc]


  def _get_gaussian_params(self, means, stdevs, num_gaussians):
    '''
    Returns the parameters of the densities in the GMM.
    '''

    with tf.variable_scope("gmm_breakdown"):
      comp_means = tf.split(means, num_gaussians, axis=1)
      comp_stdevs = tf.split(stdevs, num_gaussians, axis=1)

    # Turn the covariance values into a diagonalized matrix, then return.
    comp_covs = tf.diag(E for E in comp_stdevs) #TODO @therealjtgill diag only
                                                # works on Rank(1) tensors

    return (comp_means, comp_covs)


  def _eval_gaussians(self, values, means, stdevs, num_gaussians):
    '''
    Takes tensors of values, means, and stdevs, and returns tensors of
    gaussians parametrized by 'means' and 'stdevs' evaluated at 'values'.
    Here we assume that 'values' only contains components relevant to the
    GMM on the output.

    values -> [bs*sl, M]
    stdevs -> [bs*sl, num_gaussians*M]
    means  -> [bs*sl, num_gaussians*M]
    '''

    with tf.variable_scope("gmm_evaluation"):
      comp_means = tf.split(means, num_gaussians, axis=1)
      comp_stdevs = tf.split(stdevs, num_gaussians, axis=1)
      
      ind_gaussians = []
      for i in range(num_gaussians):
        factors = 1./(2*np.sqrt(np.pi)*comp_stdevs[i])
        #print(self.session.run([tf.shape(comp_means[i]), tf.shape(comp_stdevs[i])]))
        norms = (values - comp_means[i])/comp_stdevs[i] # Broadcast?
        exponents = -0.5*tf.tensordot(norms, norms, axes=1)
        ind_gaussians.append(factors*tf.exp(exponents))

      # You have a gaussian for each set of components of the mixture model,
      # now you just have to reduce those components into the pieces of the GMM.

      gaussians = [tf.reduce_prod(g, axis=-1) for g in ind_gaussians]

    return tf.convert_to_tensor(gaussians)

  def train_batch(self, batch_in, batch_out):
    '''
    Trains the MDN on a single batch of input.
    Returns the loss, parameters of each gaussian, and the weights associated
    with each density in the Gaussian Mixture.
    '''

    (batch_size, sequence_length, input_size) = batch_in.shape
    zero_states = self.multi_lstm_cell.zero_state(batch_size, dtype=tf.float32)

    feeds = {
      input_data: batch_in,
      output_data: batch_out
    }

    for i in range(self.num_lstm_layers):
      feeds[self.init_states[i][0]] = zero_states[i][0]
      feeds[self.init_states[i][1]] = zero_states[i][1]

    fetches = [
      self.train_op,
      self.loss,
      self.gauss_params,
      self.mix_weights
    ]

    _, loss, params, mix = self.sesion.run(fetches, feeds=feeds)
    return (loss, params, mix)


  def validate_batch(self, batch_in, batch_out):
    '''
    Runs the network on the given input batch and calculates a loss using the
    output batch. No training is performed.
    '''

    (batch_size, sequence_length, input_size) = batch_in.shape
    zero_states = self.multi_lstm_cell.zero_state(batch_size, dtype=tf.float32)

    feeds = {
      input_data: batch_in,
      output_data: batch_out
    }

    for i in range(self.num_lstm_layers):
      feeds[self.init_states[i][0]] = zero_states[i][0]
      feeds[self.init_states[i][1]] = zero_states[i][1]

    fetches = [
      self.loss,
      self.gauss_params,
      self.mix_weights
    ]

    loss, params, mix = self.sesion.run(fetches, feeds=feeds)
    return (loss, params, mix)


  def _get_mixture_sample(self, params, mix):
    '''
    Returns a single sample from the GMM defined by params and the mixture
    weights.
    Assumes that params is a list of GMM parameters with len(params) being the
    number of gaussians in the mixture.
    Assumes that mixture is a simple numpy array, where the mixture's shape is
    one-dimensional, and its size is the number of gaussians in the mixture.
    '''

    # params[0] --> means
    # params[1] --> covariance matrices
    sample = np.zeros_like(params[0][0])
    for i in range(self.num_gaussians):
      sample += mix[i]*np.random.multivariate_gaussian(params[0][i],
                                                       params[1][i])
    return sample


  def _run_once(self, input_, initial_state):
    '''
    Takes a single input, (e.g. batch_size = 1, sequence_length = 1), passes it
    to the MDN, grabs the mixture parameters and final recurrent state of the 
    MDN. Then it takes the mixture parameters and samples from them.
    The MDN returns the sampled value, other outputs, and the final recurrent
    state.
    Assumes input_.shape = [1, 1, input_size]
    '''

    #zero_states = self.multi_lstm_cell.zero_state(1, dtype=tf.float32)

    feeds = {
      input_data: input_
    }

    for i in range(self.num_lstm_layers):
      feeds[self.init_states[i][0]] = initial_states[i][0]
      feeds[self.init_states[i][1]] = initial_states[i][1]

    fetches = [
      self.mix_weights,
      self.gauss_params,
      self.stroke,
      self.last_lstm_state
    ]

    mix, params, stroke, state = self.session.run(fetches, feeds=feeds)
    mix = np.squeeze(mix)
    squeezed_params = []
    squeezed_params[0] = np.squeeze(params[0])
    squeezed_params[1] = np.squeeze(params[1])
    #stroke = np.squeeze(stroke)
    #state = np.squeeze(state)

    # Need to add a way to sample from this distribution, then return the
    # value that was sampled, and the stroke probability.
    # Weighted average of samples created from individual gaussian components?
    # np.random.multivariate_gaussian
    
    # Making some assumptions about dimensionality of the outputs here.
    # Assumptions are covered in the docstring.
    # sample = np.zeros_like(squeezed_params[0][0])
    # for i in range(self.num_gaussians):
    #   temp_sample = np.random.multivariate_gaussian(squeezed_params[0][i],
    #                                                 squeezed_params[1][i])
    #   sample += mix[i]*temp_sample
    sample = self._get_mixture_sample(squeezed_params, mix)

    return (sample, stroke, state)


  def run_cyclically(self, input_, num_steps):
    '''
    Takes a seed value, passes it to the MDN, the mixture density is sampled,
    and the sample is fed into the input of the MDN at the next timestep.
    Assumes that the shape of input_ is [1, T, input_size].
    '''

    if len(input_.shape) == 2:
      input_ = np.expand_dims(input_, axis=0)
    elif len(initial_shape) == 1:
      input_ = np.expand_dims(input_, axis=0)
      input_ = np.expand_dims(input_, axis=0)

    (batch_size, sequence_length, input_size) = input_.shape
    zero_states = self.multi_lstm_cell.zero_state(batch_size, dtype=tf.float32)

    feeds = {
      input_data: input_
    }

    for i in range(self.num_lstm_layers):
      feeds[self.init_states[i][0]] = zero_states[i][0]
      feeds[self.init_states[i][1]] = zero_states[i][1]

    fetches = [
      self.loss,
      self.mix_weights,
      self.gauss_params,
      self.stroke
      self.last_lstm_state
    ]

    loss, mix, params, stroke, state = self.session.run(fetches, feeds=feeds)

    # Need to loop over the method "_run_once" and constantly update its
    # initial recurrent state and input value.

    place = np.zeros((1, num_steps, input_size))
    stroke = np.zeros((1, num_steps, 1))
    state = np.zeros_like(zero_states)

    place[0,:] = self._get_mixture_sample()

    for i in range(1, num_steps):
      #TODO @therealjtgill: abstract the sampling procedure away into a
      # separate method, and call it here to get an initial sample.
      place[i,:], stroke[i,:], state[i,:] = self._run_once(input_, )

    return 


if __name__ == "__main__":
  stuff = MDN(tf.Session(), 3, 10)