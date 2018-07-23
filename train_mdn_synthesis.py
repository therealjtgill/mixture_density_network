from mdn_synthesis import AttentionMDN
from utils import gaussian_mixture_contour

import argparse
import copy
import datetime
import os
import scipy.misc
import sys

import data_handler as dh
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def save_weighted_means(means, weights, input_, save_dir, offset=0):
  '''
  Just in case the heat maps are too low in resolution to actually plot
  individual points, this method plots the input along with the sum of the
  weighted means from the mixture.
  '''

  sequence_length, num_gaussians, _ = weights.shape
  map_preds = []
  for i in range(sequence_length):
    pred = 0
    for j in range(num_gaussians):
      pred += weights[i, j] * means[i, j]
    map_preds.append(pred)
  map_preds = np.stack(map_preds, axis=0)

  plt.figure()
  plt.scatter(input_[:,0], -input_[:,1], s=2, color="r")
  plt.scatter(map_preds[:,0], -map_preds[:,1], s=2, color="b")
  plt.savefig(os.path.join(save_dir, "meanplot" + str(offset) + ".png"))
  plt.close()


def save_weighted_deltas(means, weights, stroke, input_, save_dir, offset=0, title=""):
  '''
  This is meant to be used with (x,y) coordinates from the input data
  representing deltas from pen position at t=i and t=i+1.
  Successive deltas are added together (for the inputs as well as the weighted
  means) to produce an image of what was actually written.
  '''

  sequence_length, num_gaussians, _ = weights.shape
  print("stroke shape:", stroke.shape)
  print("means shape:", means.shape)
  print("input_ shape:", input_.shape)
  breaks = np.where(stroke > 0.8)[0]
  breaks_ = np.where(input_[:,2] > 0.8)[0]
  print("breaks:", breaks_)
  map_preds = []
  for i in range(sequence_length):
    pred = 0
    for j in range(num_gaussians):
      pred += weights[i, j] * means[i, j]
    if i > 0:
      map_preds.append(pred + map_preds[-1])
      input_[i, :] = input_[i, :] + input_[i - 1, :]
    else:
      map_preds.append(pred)
  map_preds = np.stack(map_preds, axis=0)

  plt.figure()
  plt.title(title)
  plt.scatter(input_[:,0], -input_[:,1], s=2, color="r")
  plt.scatter(map_preds[:,0], -map_preds[:,1], s=2, color="b")
  if len(breaks_) > 0:
    for i in range(1, len(breaks_)):
      plt.plot(input_[breaks_[i-1]+1:breaks_[i],0], -input_[breaks_[i-1]+1:breaks_[i],1], color="r")
    for i in range(1, len(breaks_)):
      plt.plot(map_preds[breaks_[i-1]+1:breaks_[i],0], -map_preds[breaks_[i-1]+1:breaks_[i],1], color="b")
  else:
    plt.plot(input_[:,0], -input_[:,1], color="r")
    plt.plot(map_preds[:,0], -map_preds[:,1], color="b")
  plt.axis("scaled")
  plt.savefig(os.path.join(save_dir, "deltasplot" + str(offset) + ".png"))
  plt.close()


def save_dots(dots, strokes, save_dir, offset=0):

#  breaks = np.squeeze(np.where(data[:,2] == 1))
#  print('breaks:', breaks)
#
#  plt.figure()
#  for i in range(1, len(breaks)):
#    #print('x', data[breaks[i-1]:breaks[i], 0], 'y', data[breaks[i-1]:breaks[i], 1])
#    plt.plot(data[breaks[i-1]+1:breaks[i], 0], data[breaks[i-1]+1:breaks[i], 1])
#  plt.show()

  dots = np.squeeze(dots)
  # Got rid of the squeeze on strokes because it has then potential to maintain a shape of [1,1,1]
  breaks = np.where(strokes[0,:,0] > 0.8)[0]
  print("strokes shape:", strokes.shape)
  print("dots shape:", dots.shape)
  print("dots breaks:", breaks)
  print("strokes shape:", strokes.shape)
  sequence_length, _ = dots.shape
  map_dots = []
  for i in range(sequence_length):
    if i > 0:
      map_dots.append(dots[i,:] + map_dots[-1])
    else:
      map_dots.append(dots[i,:])
  map_dots = np.stack(map_dots, axis=0)

  plt.figure()
  #plt.scatter(map_dots[:,0], -map_dots[:,1], s=2, color="b")
  if len(breaks) > 0:
    for i in range(1, len(breaks)):
      plt.plot(map_dots[breaks[i-1]+1:breaks[i],0], -map_dots[breaks[i-1]+1:breaks[i],1], color="b")
  else:
    plt.plot(map_dots[:,0], -map_dots[:,0], color="b")
  plt.axis("scaled")
  plt.savefig(os.path.join(save_dir, "cyclicalrunplot" + str(offset) + ".png"))
  plt.close()


def save_attention_weights(att, save_dir, offset=0, ylabels=None, suffix="", title="", filename="attention_weights"):

  num_chars, alphabet_size = att.shape
  print("attention weights shape:", att.shape)
  np.savetxt(os.path.join(save_dir, "attention_weights" + suffix + str(i) + ".dat"), np.squeeze(att))
  plt.figure()
  plt.title(title)
  if not ylabels == None:
    plt.yticks(range(att.shape[0]), ylabels, fontsize=6)
    plt.rcParams['ytick.labelsize'] = 12
  plt.xlabel("sequence position")
  plt.ylabel("alphabet index")
  plt.imshow(np.squeeze(att).T, interpolation="nearest", cmap="plasma", aspect=8)
  plt.savefig(os.path.join(save_dir, filename + str(i) + ".png"), dpi=600)
  plt.close()
  np.savetxt(os.path.join(save_dir, filename + str(i) + ".dat"), np.squeeze(att))


def save_mixture_weights(weights, save_dir, offset=0, suffix="", title=""):

  sequence_length, num_gaussians, _ = weights.shape
  #print("sample weights values:", np.squeeze(weights))
  #print("  weights sum along last axis:", np.sum(np.squeeze(weights), axis=-1))
  print("mixture weights shape:", weights.shape)
  np.savetxt(os.path.join(save_dir, "mixture_weights" + suffix + str(i) + ".dat"), np.squeeze(weights))
  plt.figure()
  plt.title(title)
  plt.xlabel("sequence position")
  plt.ylabel("gaussian mixture component index")
  #plt.imsave(os.path.join(save_dir, "mixture_weights" + str(i) + ".png"), np.squeeze(weights).T, vmin=0, vmax=1, interpolation='nearest')
  plt.imshow(np.squeeze(weights).T, interpolation="nearest", cmap="gray", vmin=0.0, vmax=1.0)
  plt.savefig(os.path.join(save_dir, "mixture_weights" + str(i) + ".png"))
  plt.close()
  np.savetxt(os.path.join(save_dir, "mixture_weights" + str(i) + ".dat"), np.squeeze(weights))


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="The script used to train a mixture density network. Assumes that cleaned training data is present.")

  parser.add_argument("--traindata", action="store", dest="data_dir", type=str, required=True,
                      help="Specify the location of a training data pickle file, including the filename (usually in \"./data_clean/some_folder/data.pkl\"). \
                      This pickle file only exists if you've run data_cleaner.py on the original dataset.")
  parser.add_argument("--nummixcomps", action="store", dest="num_mix_components", type=int, default=20,
                      help="Optional argument to specify the number of gaussians in the Gaussian Mixture Model. \
                      Note that adding more components requires a greater number of weights on the output layer.")
  parser.add_argument("--numlayers", action="store", dest="num_layers", type=int, default=3,
                      help="Optional argument to specify the number of LSTM layers that will be used as part of the NN. \
                      Note that a large number of layers will decrease training error, but will require more RAM.")
  parser.add_argument("--truncatedata", action="store", dest="truncate_data", type=bool, default=False,
                      help="Optional argument to only load a portion of the training data (the first 100 files). \
                      This should be used for debugging purposes only.")
  parser.add_argument("--iterations", action="store", dest="num_iterations", type=int, default=75000,
                      help="Supply a maximum number of iterations of network training. This is the number of batches of \
                      data that will be presented to the network, NOT the number of epochs.")
  parser.add_argument("--selfcycles", action="store", dest="num_cycles", type=int, default=400,
                      help="Have the network generate its own data points for this number of timesteps.")
  parser.add_argument("--numattcomps", action="store", dest="num_att_components", type=int, default=7,
                      help="Number of attention gaussians for convolution with one-hot ASCII text.")
  parser.add_argument("--checkpointfile", action="store", dest="checkpoint_file", type=str, default=None,
                      help="Location of a checkpoint file to be loaded (for additional training or running).")

  args = parser.parse_args()

  data_dir = args.data_dir
  num_mix_components = args.num_mix_components
  num_layers = args.num_layers
  num_att_components = args.num_att_components
  checkpoint_file = args.checkpoint_file
  input_size = 3

  if not os.path.exists(data_dir):
    print("Could not find data at location: %s\nExiting." % data_dir)
    sys.exit(-1)

  data_file = data_dir

  if args.truncate_data:
    print("Using the truncated dataset.")
    dh = dh.data_handler(data_file[0:100], [.7, .15, .15])
  else:
    print("Using the full dataset.")
    dh = dh.data_handler(data_file, [.7, .15, .15])

  tf.reset_default_graph()

  session = tf.Session()
  mdn_model = AttentionMDN(session, input_size, num_att_components, num_mix_components, 250, alphabet_size=dh.alphabet_size(), save=True)
  if checkpoint_file == None:
    session.run(tf.global_variables_initializer())
  else:
    #session.run(tf.global_variables_initializer())
    print("Loading checkpoint file")
    mdn_model.load_params(checkpoint_file)
  save = tf.train.Saver()

  save_dir = os.path.expanduser("~/documents/mdn_") + str(datetime.datetime.today()).replace(":", "-").replace(" ", "-")
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  writer = tf.summary.FileWriter(save_dir, graph=session.graph)

  test_vals = dh.get_test_batch(2, 300)
  for i in range(args.num_iterations):
    print("\niteration number: ", i)
    start_time = datetime.datetime.now()
    train = dh.get_train_batch(32, 300)
    # loss, means, stdevs, mix
    things = mdn_model.train_batch(train["X"], train["onehot"], train["y"])
    #print("mixture evaluation max: ", np.amax(things[-1]), "min: ", np.amin(things[-1]))
    #print("individual gaussian evaluations max: ", np.amax(things[-2]), "min: ", np.amin(things[-2]))
    if i % 100 == 0:
      print("  saving images", i)
      temp_test_vals = copy.deepcopy(test_vals)
      things = mdn_model.validate_batch(temp_test_vals["X"], temp_test_vals["onehot"], temp_test_vals["y"])
      save_weighted_deltas(things[1][0,:,:,:], things[3][0,:,:,], things[6][0,:], temp_test_vals["y"][0,:,:], save_dir, i, title=temp_test_vals["ascii"][0])
      save_mixture_weights(things[3][0,:,:], save_dir, suffix="prediction", offset=i, title=temp_test_vals["ascii"][0])
      save_attention_weights(things[-2][0,:,:], save_dir, suffix="window", ylabels=dh.alphabet, title=temp_test_vals["ascii"][0], offset=i)
      save_attention_weights(things[-1][0,:,:], save_dir, suffix="window", ylabels=[c for c in test_vals["ascii"][0]], title=temp_test_vals["ascii"][0], offset=i, filename="phi")
      #save_weighted_deltas(things[1][0,:,:,:], things[3][0,:,:,], things[6][0,:], train["y"][0,:,:], save_dir, i, title=train["ascii"][0])
      #save_mixture_weights(things[3][0,:,:], save_dir, suffix="prediction", offset=i, title=train["ascii"][0])
      #save_attention_weights(things[-1][0,:,:], save_dir, suffix="window", ylabels=dh.alphabet, title=train["ascii"][0], offset=i)

    if i % 500 == 0:
      mdn_model.save_params(os.path.join(save_dir, "mdn_model_ckpt"), i)

    #mdn_model.validate_batch(fake_train_in, fake_train_out)

    print("loss: ", things[0], "loss shape: ", things[0].shape)
    delta = datetime.datetime.now() - start_time
    with open(os.path.join(save_dir, "_error.dat"), "a") as f:
      f.write(str(i) + "," + str(things[0]) + ", " + str(delta.total_seconds()) + "\n")
