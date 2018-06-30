from mdn import MDN
from utils import gaussian_mixture_contour

import argparse
import datetime
import os
import scipy.misc
import sys

import data_handler as dh
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def save_prediction_heatmap(means, stdevs, weights, input_, save_dir, offset=0):
  '''
  Expects means and stdevs to have the shape:
  means.shape = [sequence_length, num_gaussians, num_params]
  stdevs.shape = [sequence_length, num_gaussians, num_params]
  weights.shape = [sequence_length, num_gaussians, 1]
  '''
  x_box = (np.amin(means[:, :, 0]) - 0.25, np.amax(means[:, :, 0]) + 0.25)
  y_box = (np.amin(means[:, :, 1]) - 0.25, np.amax(means[:, :, 1]) + 0.25)

  x_grid, y_grid = np.mgrid[x_box[0]:x_box[1]:0.01,
                            y_box[0]:y_box[1]:0.01]

  #print("printout mix_weights shape:", weights.shape)
  #print("printout stdevs shape:", stdevs.shape)
  sequence_length, num_gaussians, _ = weights.shape

  z = None
  for i in range(sequence_length):
    #print("printout stdevs vals:", stdevs[i, 0, :])
    covars = [np.diag(stdevs[i, j, :]) for j in range(num_gaussians)]
    if i == 0:
      z = gaussian_mixture_contour(means[i], covars, weights[i], x_box, y_box)
    else:
      z += gaussian_mixture_contour(means[i], covars, weights[i], x_box, y_box)

  #return z
  plt.figure()
  plt.contourf(x_grid, -y_grid, z)
  plt.scatter(input_[:,0], -input_[:,1], s=2, color="r")
  plt.savefig(os.path.join(save_dir, "heatmap" + str(offset) + ".png"))
  plt.close()


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


def save_weighted_deltas(means, weights, stroke, input_, save_dir, offset=0):
  '''
  This is meant to be used with (x,y) coordinates from the input data
  representing deltas from pen position at t=i and t=i+1.
  Successive deltas are added together (for the inputs as well as the weighted
  means) to produce an image of what was actually written.
  '''

  sequence_length, num_gaussians, _ = weights.shape
  breaks = np.squeeze(np.where(np.squeeze(stroke) > 0.8))
  breaks_ = np.squeeze(np.where(input_[:,2] == 1))
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
  #plt.scatter(input_[:,0], -input_[:,1], s=2, color="r")
  #plt.scatter(map_preds[:,0], -map_preds[:,1], s=2, color="b")
  if len(breaks) > 0:
    for i in range(len(breaks)):
      plt.plot(map_preds[breaks[i-1]+1:breaks[i],0], -map_preds[breaks[i-1]+1:breaks[i],1], color="b")
  else:
    plt.plot(map_preds[:,0], -map_preds[:,1], color="b")

  if len(breaks_) > 0:
    for i in range(len(breaks_)):
      plt.plot(input_[breaks_[i-1]+1:breaks_[i],0], -input_[breaks_[i-1]+1:breaks_[i],1], color="r")
  else:
    plt.plot(input_[:,0], -input_[:,1], color="r")

  plt.savefig(os.path.join(save_dir, "deltasplot" + str(offset) + ".png"))
  plt.close()


def save_mixture_weights(weights, save_dir, offset=0):

  sequence_length, num_gaussians, _ = weights.shape
  #print("sample weights values:", np.squeeze(weights))
  #print("  weights sum along last axis:", np.sum(np.squeeze(weights), axis=-1))
  print("mixture weights shape:", weights.shape)
  np.savetxt(os.path.join(save_dir, "mixture_weights" + str(i) + ".dat"), np.squeeze(weights))
  plt.figure()
  plt.xlabel("sequence position")
  plt.ylabel("gaussian mixture component index")
  #plt.imsave(os.path.join(save_dir, "mixture_weights" + str(i) + ".png"), np.squeeze(weights).T, vmin=0, vmax=1, interpolation='nearest')
  plt.imshow(np.squeeze(weights).T, interpolation="nearest", cmap="gray", vmin=0.0, vmax=1.0)
  plt.savefig(os.path.join(save_dir, "mixture_weights" + str(i) + ".png"))
  plt.close()
  np.savetxt(os.path.join(save_dir, "mixture_weights" + str(i) + ".dat"), np.squeeze(weights))


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
  breaks = np.squeeze(np.where(np.squeeze(strokes) > 0.8))
  print("dots shape: ", dots.shape)
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
    for i in range(len(breaks)):
      plt.plot(map_dots[breaks[i-1]+1:breaks[i],0], -map_dots[breaks[i-1]+1:breaks[i],1], color="b")
  else:
    plt.plot(map_dots[:,0], -map_dots[:,0], color="b")
  plt.savefig(os.path.join(save_dir, "cyclicalrunplot" + str(offset) + ".png"))
  plt.close()


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="The script used to train a mixture density network. Assumes that cleaned training data is present.")

  parser.add_argument("--traindata", action="store", dest="data_dir", type=str, required=True,
                      help="Specify a location of training data (usually in \"./data_clean/some_folder/data.pkl\")")
  parser.add_argument("--nummixcomps", action="store", dest="num_components", type=int, default=6,
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

  args = parser.parse_args()

  data_dir = args.data_dir
  num_components = args.num_components
  num_layers = args.num_layers
  #num_data_files = args.num_data_files

  if not os.path.exists(data_dir):
    sys.exit(-1)

  session = tf.Session()
  mdn_model = MDN(session, num_layers, num_components, 250, save=True)
  session.run(tf.global_variables_initializer())
  save = tf.train.Saver()

  save_dir = os.path.expanduser("~/documents/mdn_") + str(datetime.datetime.today()).replace(":", "-").replace(" ", "-")
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  writer = tf.summary.FileWriter(save_dir, graph=session.graph)

  #data_files = os.listdir(data_dir)
  #data_files = [os.path.join(data_dir, d) for d in data_files if ".csv" in d]
  data_file = data_dir

  if args.truncate_data:
    print("Using the truncated dataset.")
    dh = dh.data_handler(data_file[0:100], [.7, .15, .15])
  else:
    print("Using the full dataset.")
    dh = dh.data_handler(data_file, [.7, .15, .15])

  for i in range(args.num_iterations):
    start_time = datetime.datetime.now()
    train = dh.get_train_batch(32, 300)
    # loss, means, stdevs, mix
    things = mdn_model.train_batch(train["X"], train["y"])
    print("mixture evaluation max: ", np.amax(things[-1]), "min: ", np.amin(things[-1]))
    print("individual gaussian evaluations max: ", np.amax(things[-2]), "min: ", np.amin(things[-2]))
    if i % 100 == 0:
      print("  saving images", i)
      #validate = dh.get_validation_batch(1, 10)
      test = dh.get_test_batch(1, 100)
      dots, strokes = mdn_model.run_cyclically(test["X"], 400)
      save_dots(dots, strokes, save_dir, i)
      #dots, strokes = mdn_model.run_cyclically(validate["X"], 400)
      #valid = mdn_model.validate_batch(validate["X"], validate["y"])
      #save_prediction_heatmap(things[1][0,:,:,:], things[2][0,:,:,:], things[3][0,:,:], train["y"][0,:,:], save_dir, i)
      save_weighted_deltas(things[1][0,:,:,:], things[3][0,:,:,], things[6][0,:], train["y"][0,:,:], save_dir, i)
      save_mixture_weights(things[3][0,:,:], save_dir, i)

    if i % 500 == 0:
      mdn_model.save_params(os.path.join(save_dir, "mdn_model"), i)

    #mdn_model.validate_batch(fake_train_in, fake_train_out)

    print("loss: ", things[0], "loss shape: ", things[0].shape)
    delta = datetime.datetime.now() - start_time
    with open(os.path.join(save_dir, "_error.dat"), "a") as f:
      f.write(str(i) + "," + str(things[0]) + ", " + str(delta.total_seconds()) + "\n")
