from mdn import MDN
import tensorflow as tf
import numpy as np
import data_handler as dh
import matplotlib.pyplot as plt
import scipy.misc
from utils import gaussian_mixture_contour
import os
import datetime


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


def save_mixture_weights(weights, save_dir, offset=0):

  sequence_length, num_gaussians, _ = weights.shape
  #print("sample weights values:", np.squeeze(weights))
  #print("  weights sum along last axis:", np.sum(np.squeeze(weights), axis=-1))
  print("mixture weights shape:", weights.shape)
  #scipy.misc.imsave(save_dir, "mixture_weights" + str(i) + ".png", np.squeeze(weights))
  plt.figure()
  plt.xlabel("sequence position")
  plt.ylabel("gaussian mixture component index")
  #plt.imsave(os.path.join(save_dir, "mixture_weights" + str(i) + ".png"), np.squeeze(weights).T, vmin=0, vmax=1, interpolation='nearest')
  plt.imshow(np.squeeze(weights).T, interpolation="nearest", cmap="gray", vmin=0.0, vmax=1.0)
  plt.savefig(os.path.join(save_dir, "mixture_weights" + str(i) + ".png"))
  plt.close()
  np.savetxt(os.path.join(save_dir, "mixture_weights" + str(i) + ".dat"), np.squeeze(weights))


if __name__ == "__main__":

  save_dir = "C:\\users\\gsaa\\documents\\mdn_" + str(datetime.datetime.today()).replace(":", "-")
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  session = tf.Session()
  mdn_model = MDN(session, 3, 20, 250)

  dh = dh.data_handler("data_clean/handwritingfull.csv", [.7, .15, .15])

  #train = dh.get_train_batch(64, 100)
  #train = np.random.rand(64, 100, 3)
  #train_out = np.random.rand(64, 100, 3)
  #things = mdn_model.train_batch(train, train_out)

  #print(things[0])
  # Can't save things inside of filestream, have to point it to a different
  # location on the physical disk.
  #mdn_model.save_params("C:\\users\\jtgil\\documents\\mdn", 1)

  for i in range(100000):
    train = dh.get_train_batch(32, 300)
    things = mdn_model.train_batch(train["X"], train["y"])

    if i % 10 == 0:
      print("  validating", i)
      validate = dh.get_validation_batch(1, 10)

      #dots, strokes = mdn_model.run_cyclically(validate["X"], 400)
      #valid = mdn_model.validate_batch(validate["X"], validate["y"])
      print("  things[3].shape (mixes):", things[3].shape)
      save_prediction_heatmap(things[1][0,:,:,:], things[2][0,:,:,:], things[3][0,:,:], train["y"][0,:,:], save_dir, i)
      save_weighted_means(things[1][0,:,:,:], things[3][0,:,:], train["y"][0,:,:], save_dir, i)
      save_mixture_weights(things[3][0,:,:], save_dir, i)

    if i % 500 == 0:
      mdn_model.save_params(os.path.join(save_dir, "mdn_model"), i)

    #mdn_model.validate_batch(fake_train_in, fake_train_out)
    #dots, strokes = mdn_model.run_cyclically(np.random.rand(1, 15, 3), 100)
    #print(dots.shape)
    #print(strokes.shape)
    print(things[0], things[0].shape)
    with open(os.path.join(save_dir, "error.dat"), "a") as f:
      f.write(str(i) + "," + str(things[0]) + "\n")
