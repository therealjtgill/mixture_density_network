import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.misc
import sys


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

  dots = np.squeeze(dots)
  # Got rid of the squeeze on strokes because it has then potential to maintain a shape of [1,1,1]
  breaks = np.where(strokes[0,:,0] > 0.4)[0]
  breaks = [0,] + breaks
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
  np.savetxt(os.path.join(save_dir, "attention_weights" + suffix + str(offset) + ".dat"), np.squeeze(att))
  plt.figure()
  plt.title(title)
  if not ylabels == None:
    plt.yticks(range(att.shape[0]), ylabels, fontsize=6)
    plt.rcParams['ytick.labelsize'] = 12
  plt.xlabel("sequence position")
  plt.ylabel("alphabet index")
  plt.imshow(np.squeeze(att).T, interpolation="nearest", cmap="jet", aspect=8)
  plt.savefig(os.path.join(save_dir, filename + str(offset) + ".png"), dpi=600)
  plt.close()
  #np.savetxt(os.path.join(save_dir, filename + str(i) + ".dat"), np.squeeze(att))


def save_mixture_weights(weights, save_dir, offset=0, suffix="", title=""):

  sequence_length, num_gaussians, _ = weights.shape
  #print("sample weights values:", np.squeeze(weights))
  #print("  weights sum along last axis:", np.sum(np.squeeze(weights), axis=-1))
  print("mixture weights shape:", weights.shape)
  #np.savetxt(os.path.join(save_dir, "mixture_weights" + suffix + str(i) + ".dat"), np.squeeze(weights))
  plt.figure()
  plt.title(title)
  plt.xlabel("sequence position")
  plt.ylabel("gaussian mixture component index")
  #plt.imsave(os.path.join(save_dir, "mixture_weights" + str(i) + ".png"), np.squeeze(weights).T, vmin=0, vmax=1, interpolation='nearest')
  plt.imshow(np.squeeze(weights).T, interpolation="nearest", cmap="gray", vmin=0.0, vmax=1.0)
  plt.savefig(os.path.join(save_dir, "mixture_weights" + str(offset) + ".png"))
  plt.close()
  #np.savetxt(os.path.join(save_dir, "mixture_weights" + str(i) + ".dat"), np.squeeze(weights))
