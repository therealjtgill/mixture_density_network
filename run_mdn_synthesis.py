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


def save_dots(dots, strokes, save_dir, offset=0, suffix=""):

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
  plt.scatter(map_dots[:,0], -map_dots[:,1], s=2, color="b")
  #plt.scatter(map_dots[:,0], -map_dots[:,1], s=2, color="b")
  if len(breaks) > 0:
    for i in range(1, len(breaks)):
      plt.plot(map_dots[breaks[i-1]+1:breaks[i],0], -map_dots[breaks[i-1]+1:breaks[i],1], color="b")
  else:
    plt.plot(map_dots[:,0], -map_dots[:,0], color="b")
  plt.axis("scaled")
  plt.savefig(os.path.join(save_dir, "cyclicalrunplot_" + str(suffix) + str(offset) + ".png"))
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


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="The script used to run a mixture density network that has been trained on handwriting and conditioned on ASCII text.")

  parser.add_argument("--traindata", action="store", dest="data_dir", type=str, required=True,
                      help="Specify the location of a training data pickle file, including the filename (usually in \"./data_clean/some_folder/data.pkl\"). \
                      This pickle file only exists if you've run data_cleaner.py on the original dataset.")
  parser.add_argument("--checkpointfile", action="store", dest="checkpoint_file", type=str, default=None,
                      help="Location of a checkpoint file to be loaded (for additional training or running).")
  parser.add_argument("--string", action="store", dest="ascii_string", type=str, required=True, default="this is a test",
                      help="The ASCII string that you want the network to write out as handwriting.")
  parser.add_argument("--numsamples", action="store", dest="num_samples", type=int, required=False, default=1,
                      help="The number of sequences of handwriting to generate from the given ASCII string. Use with a low bias value to get more diverse output.")
  parser.add_argument("--bias", action="store", dest="bias", type=float, required=False, default=0,
                      help="How heavily to bias sampling. Low/high bias values lead to consistent/varied styles of handwriting across samples.")

  args = parser.parse_args()

  data_dir = args.data_dir
  num_mix_components = 20
  num_att_components = 10
  checkpoint_file = args.checkpoint_file
  ascii_string = args.ascii_string
  num_samples = args.num_samples
  bias = args.bias
  input_size = 3

  if not os.path.exists(data_dir):
    print("Could not find data at location: %s\nExiting." % data_dir)
    sys.exit(-1)

  if checkpoint_file == None:
    print("No checkpoint file specified!\nExiting.")
    sys.exit(-1)

  data_file = data_dir

  dh = dh.data_handler(data_file, [.7, .15, .15])

  tf.reset_default_graph()

  session = tf.Session()
  mdn_model = AttentionMDN(session, input_size, num_att_components, num_mix_components, 250, alphabet_size=dh.alphabet_size(), save=True)
    
  print("Loading checkpoint file")
  mdn_model.load_params(checkpoint_file)
  save = tf.train.Saver()

  save_dir = os.path.expanduser("~/documents/mdn_run_") + str(datetime.datetime.today()).replace(":", "-").replace(" ", "-") + "_" + ascii_string
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  writer = tf.summary.FileWriter(save_dir, graph=session.graph)
  test_vals = dh.get_test_batch(1, 100)

  for i in range(num_samples):
    print("using bias: ", bias)
    dots, strokes, alphabet_weights, phi = mdn_model.run_cyclically(np.zeros((1,1,3)), np.stack([dh.ascii_to_one_hot(args.ascii_string)], axis=0), 1000, bias)
    print("alphabet_weights shape:", alphabet_weights.shape)
    save_dots(dots, strokes, save_dir, i, suffix=str(bias))
    save_attention_weights(alphabet_weights[0,:,:], save_dir, suffix="window", ylabels=dh.alphabet, title=args.ascii_string, offset=i)