import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)
import os
import sys
from window_cell import WindowCell

class AttentionMDN(object):
  '''
  Contains useful methods for training, testing, and validating a mixture
  density network.
  '''

  def __init__(self, session, input_size, num_att_gaussians=3, num_mix_gaussians=3, lstm_cell_size=300,
               alphabet_size=None, save=False, dropout=1.0, l2_penalty=0.0):
    '''
    Sets up the computation graph for the MDN.
    Bishop, et. al, use a mixture of univariate gaussians, which allows them
    to avoid futzing around with off-diagonal covariance matrix terms.
    Univariate gaussians have proven to be insufficient for prediction, so this
    model uses full covariance matrices for the mixture components.
    '''

    # TODO @therealjtgill - Need to add the multicell dynamic rnn states to a
    # class member variable for the run_once and run_cyclically methods.

    dtype = tf.float32
    self.session = session
    self.weights = []
    self.biases = []
    self.layers = []
    self.init_states = []
    self.dropout = dropout
    self.num_mix_gaussians = num_mix_gaussians
    self.num_att_gaussians = num_att_gaussians
    self.input_size = input_size
    self.l2_penalty = l2_penalty
    self.alphabet_size = alphabet_size
    
    num_means = num_mix_gaussians*(input_size - 1)
    num_variances = num_mix_gaussians*(input_size - 1)
    num_correlations = num_mix_gaussians*(1)
    output_size = num_mix_gaussians + num_means + num_variances \
                  + num_correlations + 1

    print("output size:", output_size)
    print("output size per gaussian:", (output_size - 1)/num_mix_gaussians)

    with tf.variable_scope("mdn"):

      # [batch_size, seq_length, input_size]
      self.input_data = tf.placeholder(dtype=dtype,
                                       shape=[None, None, input_size], name="batch_input")
      # [batch_size, num_chars, alphabet_size]
      self.input_ascii = tf.placeholder(dtype=dtype,
                                        shape=[None, None, alphabet_size], name="ascii_input")
      # [batch_size, seq_length, input_size]  
      self.output_data = tf.placeholder(dtype=dtype,
                                        shape=[None, None, input_size], name="batch_targets")
      # []
      self.input_mixture_bias = tf.placeholder(dtype=dtype,
                                 shape=[], name="bias")

      # Initial states for recurrent parts of the network
      self.zero_states = []
      self.init_states = []
      ph_c = tf.placeholder(dtype=dtype, shape=[None, lstm_cell_size])
      ph_h = tf.placeholder(dtype=dtype, shape=[None, lstm_cell_size])
      self.init_states.append(tf.nn.rnn_cell.LSTMStateTuple(ph_c, ph_h))

      ph_v = tf.placeholder(dtype=dtype, shape=[None, num_att_gaussians])
      self.init_states.append(ph_v)

      ph_c = tf.placeholder(dtype=dtype, shape=[None, lstm_cell_size])
      ph_h = tf.placeholder(dtype=dtype, shape=[None, lstm_cell_size])
      self.init_states.append(tf.nn.rnn_cell.LSTMStateTuple(ph_c, ph_h))

      self.init_states = tuple(self.init_states)
      # End initial states

      batch_size = tf.cast(tf.shape(self.input_data)[0], tf.int32)
      seq_length = tf.cast(tf.shape(self.input_data)[1], tf.int32)
      num_chars = tf.cast(tf.shape(self.input_ascii)[1], tf.int32)

      # The attention mechanism from the paper requires an LSTM above it. Only
      # one of the parameters of the attention mechanism is recurrent, which
      # means that we need to split the output of the LSTM between recurrent
      # and non-recurrent output.

      # Attention mechanism
      self.recurrent_states = []
      lstm_1 = tf.nn.rnn_cell.BasicLSTMCell(lstm_cell_size)
      lstm_1_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm_1, output_keep_prob=self.dropout)
      window = WindowCell(lstm_cell_size, num_chars, self.num_att_gaussians)
      self.zero_states.append(lstm_1.zero_state(batch_size, dtype=dtype))
      self.zero_states.append(window.zero_state(batch_size, dtype=dtype))
      # Get back num_att_gaussians tensors of shape [bs, sl, 3], which is a
      # tensor of attention window parameters for each attention gaussian.
      # size(attention_params) = [num_att_gaussians, bs, sl, 3]
      #   (technically a list of size 'num_att_gaussians' with [bs, sl, 3] size 
      #   tensors at each element of the list)

      lstm_1_out, lstm_1_state = tf.nn.dynamic_rnn(lstm_1_dropout, self.input_data, dtype=dtype,
        initial_state=self.init_states[0])
      self.layers.append(lstm_1_out)
      self.recurrent_states.append(lstm_1_state)

      #self.phi, attention_state = tf.nn.dynamic_rnn(window, lstm_1_out, dtype=dtype,
      #  initial_state=self.init_states[1])
      #self.layers.append(self.phi)
      self.phi_plus, attention_state = tf.nn.dynamic_rnn(window, lstm_1_out, dtype=dtype,
        initial_state=self.init_states[1])
      self.phi, self.stop_check = tf.split(self.phi_plus, [num_chars, 1], axis=-1)
      self.layers.append(self.phi)
      self.recurrent_states.append(attention_state)
      # Need to evaluate each gaussian at an index value corresponding to a
      # character position in the ASCII input (axis=1).
      # Tile the parameters on the last axis by the number of characters in
      # the ascii sequence to ensure proper broadcasting. Need to get a
      # tensor with the following shape for phi
      #   [bs, sl, nc] (nc = num_chars)
      # and a tensor with the following shape for the soft window weight.
      #   [bs, sl, as] (as = alphabet_size)

      # shape(self.phi) = [bs, sl, nc]
      # shape(ascii_input) = [bs, nc, as]
      # shape(self.alphabet_weights) = [bs, sl, as]
      self.alphabet_weights = tf.matmul(self.phi, self.input_ascii)
      self.layers.append(self.alphabet_weights)
      # End attention mechanism

      # Final recurrent layer
      lstm_2 = tf.nn.rnn_cell.BasicLSTMCell(lstm_cell_size, name="a")
      lstm_2_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm_2, output_keep_prob=self.dropout)

      lstm_2_input = tf.concat([self.alphabet_weights, self.input_data, lstm_1_out], axis=-1)
      last_lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_2_dropout,])
      outputs, outputs_state = \
        tf.nn.dynamic_rnn(last_lstm_cells, lstm_2_input, dtype=dtype,
                          initial_state=self.init_states[2:3])
      outputs_flat = tf.reshape(outputs, [-1, lstm_cell_size], name="dynamic_rnn_reshape")
      self.recurrent_states.append(outputs_state)
      self.layers.append(outputs_flat)
      self.zero_states.append(lstm_2.zero_state(batch_size, dtype=dtype))
      # End final recurrent layer

      # Output layer
      shape = [lstm_cell_size, output_size]
      self.layers.append(self._linear_op(self.layers[-1], shape))

      # Get the mixture components
      splits = [num_means, num_variances, num_correlations, num_mix_gaussians, 1]
      pieces = tf.split(self.layers[-1], splits, axis=1)
      self.mixture_bias = tf.nn.relu(self.input_mixture_bias)
      self.means = pieces[0]
      self.stdevs = tf.exp(pieces[1] - self.mixture_bias)
      self.correls = tf.nn.tanh(pieces[2])
      self.mix_weights = tf.nn.softmax(pieces[3]*(1 + self.mixture_bias))
      self.stroke = tf.nn.sigmoid(pieces[4])

      # Reshape the means, stdevs, correlations, and mixture weights for
      # friendly returns
      means_shape = [batch_size, seq_length, num_mix_gaussians, 2]
      stdevs_shape = [batch_size, seq_length, num_mix_gaussians, 2]
      mixes_shape = [batch_size, seq_length, num_mix_gaussians, 1]
      correls_shape = [batch_size, seq_length, num_mix_gaussians, 2]
      self.means_ = tf.reshape(self.means, means_shape)
      self.stdevs_ = tf.reshape(self.stdevs, stdevs_shape)
      self.mix_weights_ = tf.reshape(self.mix_weights, mixes_shape)
      self.correls_ = tf.reshape(self.correls, correls_shape)

      outputs_flat = tf.reshape(self.output_data, [-1, input_size])

      gauss_values, stroke = tf.split(outputs_flat, [input_size-1, 1], axis=1)

      # Grab these for sampling from the network.
      self.gauss_params = \
        self._get_gaussian_params(self.means, self.stdevs, self.correls,
                                  num_mix_gaussians)

      # These are for training or evaluating the network.
      self.gauss_evals = self._eval_gaussians(gauss_values, self.means,
                                              self.stdevs, self.correls,
                                              num_mix_gaussians)

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
      optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, momentum=0.9)
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


  def _get_gaussian_params(self, means, stdevs, correls, num_mix_gaussians):
    '''
    Returns the parameters of the densities in the GMM.
    '''

    with tf.variable_scope("gmm_breakdown"):
      comp_means = tf.split(means, num_mix_gaussians, axis=1)
      comp_stdevs = tf.split(stdevs, num_mix_gaussians, axis=1)
      comp_correls = tf.split(correls, num_mix_gaussians, axis=1)

    return (comp_means, comp_stdevs, comp_correls)


  def _eval_gaussians(self, values, means, stdevs, correls, num_mix_gaussians):
    '''
    Takes tensors of values, means, and stdevs, and returns tensors of
    gaussians parametrized by 'means', 'stdevs', and 'correls' evaluated at
    'values'. Here we assume that 'values' only contains components relevant
    to the GMM on the output.

    values  -> [bs*sl, M]
    stdevs  -> [bs*sl, num_mix_gaussians*M]
    means   -> [bs*sl, num_mix_gaussians*M]
    correls -> [bs*sl, num_mix_gaussians]
    '''

    print("gaussian component shapes:")
    print("\tvalues:", values.shape)
    print("\tstdevs:", stdevs.shape)
    print("\tmeans:", means.shape)
    print("\tcorrels:", correls.shape)

    with tf.variable_scope("gmm_evaluation"):
      comp_means = tf.split(means, num_mix_gaussians, axis=1)
      comp_stdevs = tf.split(stdevs, num_mix_gaussians, axis=1)
      comp_correls = tf.split(correls, num_mix_gaussians, axis=1)
      
      gaussians = []
      for i in range(num_mix_gaussians):
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
    for i in range(self.num_mix_gaussians):
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


  def train_batch(self, batch_in, batch_one_hots, batch_out):
    '''
    Trains the MDN on a single batch of input.
    Returns the loss, parameters of each gaussian, and the weights associated
    with each density in the Gaussian Mixture.
    '''

    (batch_size, sequence_length, input_size) = batch_in.shape
    #zero_states_ = self.multi_lstm_cell.zero_state(batch_size, dtype=dtype)
    #zero_states = self.session.run(zero_states_)

    feeds = {
      self.input_data: batch_in,
      self.output_data: batch_out,
      self.input_ascii: batch_one_hots,
      self.input_mixture_bias: 0
    }

    zero_states = self.session.run(self.zero_states, feed_dict=feeds)

    feeds[self.init_states[0][0]] = zero_states[0][0]
    feeds[self.init_states[0][1]] = zero_states[0][1]
    feeds[self.init_states[1]] = zero_states[1]
    feeds[self.init_states[2][0]] = zero_states[2][0]
    feeds[self.init_states[2][1]] = zero_states[2][1]

    fetches = [
      self.train_op,
      self.loss,
      self.gauss_evals,
      self.mixture,
      self.means_,
      self.stdevs_,
      self.mix_weights_,
      self.stroke,
      self.gauss_params,
      self.alphabet_weights,
      self.phi
    ]

    _, loss, gauss_eval, mix_eval, means_, stdevs_, mix, stroke, params, aw, phi = self.session.run(fetches, feed_dict=feeds)
    #print("shape of means:", means_.shape)
    #print("shape of stdevs:", stdevs_.shape)
    #print("attention_params for first batch:", atp[0, :, :])
    #print("atp shape:", atp.shape)
    correls = params[2]
    max_correl = 0
    for i in range(self.num_mix_gaussians):
      max_correl = max(max_correl, np.amax(np.sum((correls[i]*correls[i]), axis=1)))
    #print("max_correl denom:", max_correl)
    if max_correl > 1:
      print("OUT OF BOUNDS VALUE FOR MAX_CORREL")
      sys.exit(-1)
    if loss == np.nan:
      print("LOSS IS NAN. ABORTING.")
      sys.exit(-1)
    return (loss, means_, stdevs_, mix, gauss_eval, mix_eval, stroke, aw, phi)


  def validate_batch(self, batch_in, batch_one_hots, batch_out):
    '''
    Runs the network on the given input batch and calculates a loss using the
    output batch. No training is performed.
    '''

    (batch_size, sequence_length, input_size) = batch_in.shape
    #zero_states_ = self.multi_lstm_cell.zero_state(batch_size, dtype=dtype)
    #zero_states = self.session.run(zero_states_)
    #zero_states = self.session.run(self.zero_states)

    feeds = {
      self.input_data: batch_in,
      self.output_data: batch_out,
      self.input_ascii: batch_one_hots,
      self.input_mixture_bias: 0
    }

    zero_states = self.session.run(self.zero_states, feed_dict=feeds)

    feeds[self.init_states[0][0]] = zero_states[0][0]
    feeds[self.init_states[0][1]] = zero_states[0][1]
    feeds[self.init_states[1]] = zero_states[1]
    feeds[self.init_states[2][0]] = zero_states[2][0]
    feeds[self.init_states[2][1]] = zero_states[2][1]

    fetches = [
      self.loss,
      self.gauss_evals,
      self.mixture,
      self.means_,
      self.stdevs_,
      self.mix_weights_,
      self.stroke,
      self.gauss_params,
      self.alphabet_weights,
      self.phi
    ]

    loss, gauss_eval, mix_eval, means_, stdevs_, mix, stroke, params, aw, phi  = self.session.run(fetches, feed_dict=feeds)
    return (loss, means_, stdevs_, mix, gauss_eval, mix_eval, stroke, aw, phi)


  def _run_once(self, input_, stroke_, initial_states, ascii, bias):
    '''
    Takes a single input, (e.g. batch_size = 1, sequence_length = 1), passes it
    to the MDN, grabs the mixture parameters and final recurrent state of the 
    MDN. Then it takes the mixture parameters and samples from them.
    The MDN returns the sampled value, other outputs, and the final recurrent
    state.
    Assumes input_.shape = [1, 1, input_size - 1]
    Assumes stroke_.shape = [1, 1, 1]
    '''

    #zero_states = self.multi_lstm_cell.zero_state(1, dtype=dtype)
    #zero_states = self.session.run(self.zero_states)
    #print('run_once input and stroke shapes:', input_.shape, stroke_.shape)

    # Concatenate stroke and (dx, dy) input to get (1, 1, 3) input tensor.
    feeds = {
      self.input_data: np.concatenate([input_, stroke_], axis=-1),
      self.input_ascii: ascii,
      self.mixture_bias: bias
    }

    #print("len initial states[2]:", len(initial_states[2])) 

    # Initialize recurrent states with the states from the previous timestep.
    feeds[self.init_states[0][0]] = initial_states[0][0]
    feeds[self.init_states[0][1]] = initial_states[0][1]
    feeds[self.init_states[1]] = initial_states[1]
    feeds[self.init_states[2][0]] = initial_states[2][0][0] # Ugly because of multirnncell
    feeds[self.init_states[2][1]] = initial_states[2][0][1] # Ugly because of multirnncell
    # The two lines above shouldn't have an extra [0] iterator, but it'll have to
    # stay until you get rid of the multirnncell on the last LSTM layer.

    fetches = [
      self.mix_weights,
      self.gauss_params,
      self.stroke,
      self.recurrent_states,
      self.alphabet_weights,
      self.phi
    ]
    #print('input_ shape:', input_.shape)

    mix, params, stroke, state, aw, phi = self.session.run(fetches, feed_dict=feeds)
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
    return (pos_sample, stroke_sample, state, squeezed_params, mix, aw, phi)


  def run_cyclically(self, input_, ascii, num_steps, bias):
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
    #zero_states_ = self.multi_lstm_cell.zero_state(batch_size, dtype=dtype)
    #zero_states = self.session.run(zero_states_)
    #zero_states = self.session.run(self.zero_states)

    feeds = {
      self.input_data: input_,
      self.input_ascii: ascii,
      self.input_mixture_bias: bias
    }

    zero_states = self.session.run(self.zero_states, feed_dict=feeds)
    print("len zero states:", len(zero_states))
    print("len init states:", len(self.init_states))

    feeds[self.init_states[0][0]] = zero_states[0][0]
    feeds[self.init_states[0][1]] = zero_states[0][1]
    feeds[self.init_states[1]] = zero_states[1]
    feeds[self.init_states[2][0]] = zero_states[2][0]
    feeds[self.init_states[2][1]] = zero_states[2][1]

    fetches = [
      self.mix_weights,
      self.gauss_params,
      self.stroke,
      self.recurrent_states,
      self.alphabet_weights,
      self.phi
    ]

    mix, params, stroke, init_state, aw, phi = \
      self.session.run(fetches, feed_dict=feeds)
    print("mix shape:", mix.shape)
    print("params shape:", len(params), len(params[1]), params[1][0].shape)
    print("stroke shape:", stroke.shape)
    print("state lens:", [len(s) for s in init_state])
    print("init state[2][0] len", len(init_state[2][0]))
    print("type init_state[2][0]", type(init_state[2][0]))

    # Need to loop over the method "_run_once" and constantly update its
    # initial recurrent state and input value.
    dots = []
    strokes = []
    init_means = []
    init_covs = []
    init_correls = []
    phis = [phi,]
    alphabet_weights = [aw,]
    states = []
    state = np.zeros_like(zero_states)

    for j in range(self.num_mix_gaussians):
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
      temp_dot, temp_stroke, temp_state, params_, mix_, aw, phi = \
        self._run_once(dots[i-1], strokes[i-1], states[i-1], ascii, bias)
      dots.append(temp_dot)
      strokes.append(temp_stroke[np.newaxis,:,:])
      #state = init_state
      #init_state = state
      states.append(temp_state)
      alphabet_weights.append(aw)
      phis.append(phi)

    return (np.concatenate(dots, axis=1), np.concatenate(strokes, axis=1), np.concatenate(alphabet_weights, axis=1), np.concatenate(phis, axis=1)) # Getting shapes with three items: (1, sl, 2)


  def save_params(self, location, global_step):
    '''
    Simple call to save all of the parameters in the model.
    '''

    self.saver.save(self.session, location, global_step=global_step)
    #self.saver.save(self.session, location)


  def load_params(self, checkpoint_file):
    '''
    The checkpoint_file is filename.ckpt, which contains values of the model's
    trained parameters.
    The meta_file is filename.meta <-- don't need to import a meta graph, we
    already know what the model architecture is.
    '''

    #if not os.path.isfile(checkpoint_file):
    #  print("The checkpoint file", checkpoint_file, "could not be found.")
    #  return

    print("Loading checkpoint file", checkpoint_file, "into the MDN model.")
    #self.saver = tf.train.restore(self.session, checkpoint_file)
    self.saver.restore(self.session, checkpoint_file)


if __name__ == "__main__":
  
  print("This is just the script that contains the MDN class. Go away!")
