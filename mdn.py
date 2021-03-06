import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
import data_handler as dh
import os
import sys

class MDN(object):
  '''
  Contains useful methods for training, testing, and validating a mixture
  density network.
  '''

  def __init__(self, session, input_size, num_gaussians=3, num_lstm_cells=300,
               save=False):
    '''
    Sets up the computation graph for the MDN.
    Bishop, et. al, use a mixture of univariate gaussians, which allows them
    to avoid futzing around with off-diagonal covariance matrix terms.
    Univariate gaussians have proven to be insufficient for prediction, so this
    model uses full covariance matrices for the mixture components.
    '''

    dtype = tf.float32
    self.session = session
    self.weights = []
    self.biases = []
    self.layers = []
    self.init_states = []
    self.num_lstm_layers = 3
    self.num_gaussians = num_gaussians
    self.input_size = input_size
    self.l2_penalty = 0.00001
    
    num_means = num_gaussians*(input_size - 1)
    num_variances = num_gaussians*(input_size - 1)
    num_correlations = num_gaussians*(1)
    output_size = num_gaussians + num_means + num_variances \
                  + num_correlations + 1

    print("output size:", output_size)
    print("output size per gaussian:", (output_size - 1)/num_gaussians)

    with tf.variable_scope("mdn"):

      self.input_data = tf.placeholder(dtype=dtype,
                                       shape=[None, None, input_size], name="batch_input")
      self.output_data = tf.placeholder(dtype=dtype,
                                        shape=[None, None, input_size], name="batch_targets")
      batch_size = tf.shape(self.input_data)[0]
      seq_length = tf.shape(self.input_data)[1]
      print("batch_size info: ", batch_size)

      # For each layer of lstm's, create a set of placeholders to contain
      # values passed to each lstm cell's initial recurrent state.
      for i in range(self.num_lstm_layers):
        ph_c = tf.placeholder(dtype=dtype, shape=[None, num_lstm_cells])
        ph_h = tf.placeholder(dtype=dtype, shape=[None, num_lstm_cells])
        self.init_states.append(
          tf.nn.rnn_cell.LSTMStateTuple(ph_c, ph_h))

      self.init_states = tuple(self.init_states)

      lstm_layers = []
      for i in range(self.num_lstm_layers):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_lstm_cells)
        if i < self.num_lstm_layers - 1:
          self.layers.append(lstm_cell)
          lstm_layers.append(self.layers[-1])
        else:
          self.layers.append(tf.contrib.rnn.ResidualWrapper(lstm_cell))
          lstm_layers.append(self.layers[-1])

      # Get a list of LSTM cells in the current set of layers, then pass those
      # to the MultiRNNCell method.
      self.multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layers)

      # LSTM layers
      outputs, self.last_lstm_state = \
        tf.nn.dynamic_rnn(self.multi_lstm_cell, self.input_data, dtype=dtype,
                          initial_state=self.init_states)
      self.zero_states = self.multi_lstm_cell.zero_state(batch_size, dtype=dtype)
      outputs_flat = tf.reshape(outputs, [-1, num_lstm_cells], name="dynamic_rnn_reshape")
      self.layers.append(outputs_flat)

      # Output layer
      shape = [num_lstm_cells, output_size]
      self.layers.append(self._linear_op(self.layers[-1], shape))

      # Get the mixture components
      splits = [num_means, num_variances, num_correlations, num_gaussians, 1]
      pieces = tf.split(self.layers[-1], splits, axis=1)
      self.means = pieces[0]
      self.stdevs = tf.nn.softplus(pieces[1])
      self.correls = 0.9999*tf.nn.tanh(pieces[2])
      self.mix_weights = tf.nn.softmax(pieces[3])
      self.stroke = tf.nn.sigmoid(pieces[4])

      # Reshape the means, stdevs, correlations, and mixture weights for
      # friendly returns
      means_shape = [batch_size, seq_length, num_gaussians, 2]
      stdevs_shape = [batch_size, seq_length, num_gaussians, 2]
      mixes_shape = [batch_size, seq_length, num_gaussians, 1]
      correls_shape = [batch_size, seq_length, num_gaussians, 2]
      self.means_ = tf.reshape(self.means, means_shape)
      self.stdevs_ = tf.reshape(self.stdevs, stdevs_shape)
      self.mix_weights_ = tf.reshape(self.mix_weights, mixes_shape)
      self.correls_ = tf.reshape(self.correls, correls_shape)

      outputs_flat = tf.reshape(self.output_data, [-1, input_size])

      gauss_values, stroke = tf.split(outputs_flat, [input_size-1, 1], axis=1)

      # Grab these for sampling from the network.
      self.gauss_params = \
        self._get_gaussian_params(self.means, self.stdevs, self.correls,
                                  num_gaussians)

      # These are for training or evaluating the network.
      self.gauss_evals = self._eval_gaussians(gauss_values, self.means,
                                              self.stdevs, self.correls,
                                              num_gaussians)

      print(self.gauss_evals.shape, self.mix_weights.shape)
      print(pieces[3].shape)

      self.mixture = tf.reduce_sum(self.gauss_evals*self.mix_weights, axis=-1)
      stroke_loss = \
        tf.nn.sigmoid_cross_entropy_with_logits(labels=stroke, logits=pieces[4])
      print("unreduced stroke loss shape:", stroke_loss.shape)
      #a = self.gauss_evals*self.mix_weights
      #print("unreduced mixture shape:", a.shape)
      self.stroke_loss = tf.reduce_sum(stroke_loss, axis=-1)
      print(self.stroke_loss.shape)

      self.loss = tf.reduce_mean(-1*tf.log(self.mixture + 1e-8) + self.stroke_loss, name="loss")
      self.loss += self.l2_penalty*tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
      # Need to clip gradients (?)
      optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0004)
      self.train_op = optimizer.minimize(self.loss)

      if save:
        self.saver = tf.train.Saver()


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


  def _get_gaussian_params(self, means, stdevs, correls, num_gaussians):
    '''
    Returns the parameters of the densities in the GMM.
    '''

    with tf.variable_scope("gmm_breakdown"):
      comp_means = tf.split(means, num_gaussians, axis=1)
      comp_stdevs = tf.split(stdevs, num_gaussians, axis=1)
      comp_correls = tf.split(correls, num_gaussians, axis=1)

    return (comp_means, comp_stdevs, comp_correls)


  def _eval_gaussians(self, values, means, stdevs, correls, num_gaussians):
    '''
    Takes tensors of values, means, and stdevs, and returns tensors of
    gaussians parametrized by 'means', 'stdevs', and 'correls' evaluated at
    'values'. Here we assume that 'values' only contains components relevant
    to the GMM on the output.

    values  -> [bs*sl, M]
    stdevs  -> [bs*sl, num_gaussians*M]
    means   -> [bs*sl, num_gaussians*M]
    correls -> [bs*sl, num_gaussians]
    '''

    print("gaussian component shapes:")
    print("\tvalues:", values.shape)
    print("\tstdevs:", stdevs.shape)
    print("\tmeans:", means.shape)
    print("\tcorrels:", correls.shape)

    with tf.variable_scope("gmm_evaluation"):
      comp_means = tf.split(means, num_gaussians, axis=1)
      comp_stdevs = tf.split(stdevs, num_gaussians, axis=1)
      comp_correls = tf.split(correls, num_gaussians, axis=1)
      
      gaussians = []
      for i in range(num_gaussians):
        correls_denom = tf.reduce_sum(1 - comp_correls[i]*comp_correls[i], axis=1)
        factor = 1./(2*np.pi*tf.reduce_prod(comp_stdevs[i], axis=1)*tf.sqrt(correls_denom) + 1e-8)
        print("\tfactor", i, ":", factor.shape)
        #print(self.session.run([tf.shape(comp_means[i]), tf.shape(comp_stdevs[i])]))
        norms = (values - comp_means[i])/(comp_stdevs[i] + 1e-8)
        exponents = -(1/(2*correls_denom + 1e-8))*(tf.reduce_sum(norms*norms, axis=1) - tf.reduce_prod(norms, axis=1)*2*tf.reduce_sum(comp_correls[i], axis=1))
        print("\texponents", i, ":", exponents.shape)
        #ind_gaussians.append(factors*tf.exp(exponents))
        gaussians.append(factor*tf.exp(exponents))

      # You have a gaussian for each set of components of the mixture model,
      # now you just have to reduce those components into the pieces of the GMM.

      #gaussians = [tf.reduce_prod(g, axis=-1) for g in ind_gaussians]
      stacked_gaussians = tf.stack(gaussians, axis=1)
      print("stacked gaussians shape:", stacked_gaussians.shape)

    return stacked_gaussians


  def _get_mixture_sample(self, params, mix):
    '''
    Returns a single sample from the GMM defined by params and the mixture
    weights.
    Assumes that 'params' is a list of GMM parameters.
    Assumes that 'mix' is a simple numpy array, where the mixture's shape is
    one-dimensional, and its size is the number of gaussians in the mixture.
    '''

    # params[0] --> means
    # params[1] --> variance terms
    # params[2] --> correlation terms
    # Variance terms aren't in matrix form, but we know what a 2D gaussian
    # with correlated variables looks like, so we use this form to construct
    # a 2D gaussian by filling in the covariance matrix and means.

    sample = np.zeros_like(params[0][0])
    for i in range(self.num_gaussians):
      mean = params[0][i]
      #print("  mixture_sample mean shape:", mean.shape)
      cov = np.zeros((self.input_size - 1, self.input_size - 1))
      for j in range(self.input_size - 1):
        #print("  mixture_sample cov shape:", params[1][i].shape)
        cov[j,j] = params[1][i][j]
        #cov[j,1-j] = params[2][i][0]
        cov[j,1-j] = params[2][i] # Zero probably removed by squeeze operation
      #print("covariance: ", cov)
      sample += mix[i]*np.random.multivariate_normal(mean, cov)
    return sample[np.newaxis, np.newaxis, :]


  def train_batch(self, batch_in, batch_out):
    '''
    Trains the MDN on a single batch of input.
    Returns the loss, parameters of each gaussian, and the weights associated
    with each density in the Gaussian Mixture.
    '''

    (batch_size, sequence_length, input_size) = batch_in.shape

    feeds = {
      self.input_data: batch_in,
      self.output_data: batch_out
    }

    zero_states = self.session.run(self.zero_states, feed_dict=feeds)

    for i in range(self.num_lstm_layers):
      feeds[self.init_states[i][0]] = zero_states[i][0]
      feeds[self.init_states[i][1]] = zero_states[i][1]

    fetches = [
      self.train_op,
      self.loss,
      self.gauss_evals,
      self.mixture,
      self.means_,
      self.stdevs_,
      self.mix_weights_,
      self.stroke,
      self.gauss_params
    ]

    _, loss, gauss_eval, mix_eval, means_, stdevs_, mix, stroke, params = self.session.run(fetches, feed_dict=feeds)
    print("shape of means:", means_.shape)
    print("shape of stdevs:", stdevs_.shape)
    correls = params[2]
    max_correl = 0
    for i in range(self.num_gaussians):
      max_correl = max(max_correl, np.amax(np.sum((correls[i]*correls[i]), axis=1)))
    print("max_correl denom:", max_correl)
    if max_correl > 1:
      print("OUT OF BOUNDS VALUE FOR MAX_CORREL")
      sys.exit(-1)
    if loss == np.nan:
      print("LOSS IS NAN")
      sys.exit(-1)
    return (loss, means_, stdevs_, mix, gauss_eval, mix_eval, stroke)


  def validate_batch(self, batch_in, batch_out):
    '''
    Runs the network on the given input batch and calculates a loss using the
    output batch. No training is performed.
    '''

    (batch_size, sequence_length, input_size) = batch_in.shape

    feeds = {
      self.input_data: batch_in,
      self.output_data: batch_out
    }

    zero_states = self.session.run(self.zero_states, feed_dict=feeds)

    for i in range(self.num_lstm_layers):
      feeds[self.init_states[i][0]] = zero_states[i][0]
      feeds[self.init_states[i][1]] = zero_states[i][1]

    fetches = [
      self.loss,
      self.means_,
      self.stdevs_,
      self._mix_weights
    ]

    loss, means_, stdevs_, mix = self.session.run(fetches, feed_dict=feeds)
    return (loss, means_, stdevs_, mix)


  def _run_once(self, input_, stroke_, initial_states):
    '''
    Takes a single input, (e.g. batch_size = 1, sequence_length = 1), passes it
    to the MDN, grabs the mixture parameters and final recurrent state of the 
    MDN. Then it takes the mixture parameters and samples from them.
    The MDN returns the sampled value, other outputs, and the final recurrent
    state.
    Assumes input_.shape = [1, 1, input_size - 1]
    Assumes stroke_.shape = [1, 1, 1]
    '''

    #print('run_once input and stroke shapes:', input_.shape, stroke_.shape)

    # Concatenate stroke and (dx, dy) input to get (1, 1, 3) input tensor.
    feeds = {
      self.input_data: np.concatenate([input_, stroke_], axis=-1)
    }

    # Initialize recurrent states with the states from the previous timestep.
    for i in range(self.num_lstm_layers):
      feeds[self.init_states[i][0]] = initial_states[i][0]
      feeds[self.init_states[i][1]] = initial_states[i][1]

    fetches = [
      self.mix_weights,
      self.gauss_params,
      self.stroke,
      self.last_lstm_state
    ]
    #print('input_ shape:', input_.shape)

    mix, params, stroke, state = self.session.run(fetches, feed_dict=feeds)
    mix = np.squeeze(mix)
    squeezed_params = []
    squeezed_params.append([np.squeeze(p) for p in params[0]])
    squeezed_params.append([np.squeeze(p) for p in params[1]])
    squeezed_params.append([np.squeeze(p) for p in params[2]])

    # Need to add a way to sample from this distribution, then return the
    # value that was sampled, and the stroke probability.
    # np.random.multivariate_normal, et voila.
    
    # Assumptions about dimensionality of the outputs are covered in the
    # docstring.
    pos_sample = self._get_mixture_sample(squeezed_params, mix)
    stroke_sample = np.random.binomial(1, stroke)

    #return (sample, stroke, state)
    return (pos_sample, stroke_sample, state, squeezed_params, mix)


  def run_cyclically(self, input_, num_steps):
    '''
    Takes a seed value, passes it to the MDN, the mixture density is sampled,
    and the sample is fed into the input of the MDN at the next timestep.
    Assumes that the shape of input_ is [1, T, input_size].
    '''

    if len(input_.shape) == 2:
      input_ = np.expand_dims(input_, axis=0)
    elif len(input_.shape) == 1:
      input_ = np.expand_dims(input_, axis=0)
      input_ = np.expand_dims(input_, axis=0)
    print("run_cyclically input_ shape:", input_.shape)

    (batch_size, sequence_length, input_size) = input_.shape

    feeds = {
      self.input_data: input_
    }

    zero_states = self.session.run(self.zero_states, feed_dict=feeds)

    for i in range(self.num_lstm_layers):
      feeds[self.init_states[i][0]] = zero_states[i][0]
      feeds[self.init_states[i][1]] = zero_states[i][1]

    fetches = [
      self.mix_weights,
      self.gauss_params,
      self.stroke,
      self.last_lstm_state
    ]

    mix, params, stroke, init_state = \
      self.session.run(fetches, feed_dict=feeds)
    print("mix shape:", mix.shape)
    print("params shape:", len(params), len(params[1]), params[1][0].shape)
    print("stroke shape:", stroke.shape)

    # Need to loop over the method "_run_once" and constantly update its
    # initial recurrent state and input value.
    dots = []
    strokes = []
    init_means = []
    init_covs = []
    init_correls = []
    states = []
    state = np.zeros_like(zero_states)

    for j in range(self.num_gaussians):
      init_means.append(params[0][j][-1,:])
      init_covs.append(params[1][j][-1,:])
      init_correls.append(params[2][j][-1,:])

    sample = self._get_mixture_sample([init_means, init_covs, init_correls], mix[-1,:])
    print("sample shape:", sample.shape)
    dots.append(sample)
    strokes.append(stroke[np.newaxis, np.newaxis,-1,:])
    states.append(init_state)

    # Just need to stretch the dimensions of the tensors being fed back in...
    for i in range(1, num_steps):
      temp_dot, temp_stroke, temp_state, params_, mix_ = \
        self._run_once(dots[i-1], strokes[i-1], states[i-1])
      dots.append(temp_dot)
      strokes.append(temp_stroke[np.newaxis,:,:])
      #state = init_state
      #init_state = state
      states.append(temp_state)

    return (np.concatenate(dots, axis=1), np.concatenate(strokes, axis=1)) # Getting shapes with three items: (1, sl, 2)


  def save_params(self, location, global_step):
    '''
    Simple call to save all of the parameters in the model.
    '''

    self.saver.save(self.session, location, global_step=global_step)


  def load_params(self, checkpoint_file):
    '''
    The checkpoint_file is filename.ckpt, which contains values of the model's
    trained parameters.
    The meta_file is filename.meta <-- don't need to import a meta graph, we
    already know what the model architecture is.
    '''

    if not os.path.isfile(checkpoint_file):
      print("The checkpoint file", checkpoint_file, "could not be found.")
      return

    print("Loading checkpoing file", checkpoint_file, "into the MDN model.")
    self.saver = tf.train.restore(self.session, checkpoint_file)


if __name__ == "__main__":
  
  print("This is just the script that contains the MDN class! Go away.")
