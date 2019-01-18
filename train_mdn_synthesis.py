from mdn_synthesis import AttentionMDN
from utils import gaussian_mixture_contour
from plot import *

import argparse
import copy
import datetime
import os
import sys

import data_handler as dh
import numpy as np
import tensorflow as tf


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="The script used to train a mixture density network. Assumes that cleaned training data is present.")

  parser.add_argument("--traindata", action="store", dest="data_dir", type=str, required=True,
                      help="Specify the location of a training data pickle file, including the filename (usually in \"./data_clean/some_folder/data.pkl\"). \
                      This pickle file only exists if you've run data_cleaner.py on the original dataset.")
  parser.add_argument("--nummixcomps", action="store", dest="num_mix_components", type=int, default=20,
                      help="Optional argument to specify the number of gaussians in the Gaussian Mixture Model. \
                      Note that adding more components requires a greater number of weights on the output layer.")
  parser.add_argument("--truncatedata", action="store", dest="truncate_data", type=bool, default=False,
                      help="Optional argument to only load a portion of the training data (the first 100 files). \
                      This should be used for debugging purposes only.")
  parser.add_argument("--iterations", action="store", dest="num_iterations", type=int, default=75000,
                      help="Supply a maximum number of iterations of network training. This is the number of batches of \
                      data that will be presented to the network, NOT the number of epochs.")
  parser.add_argument("--numattcomps", action="store", dest="num_att_components", type=int, default=7,
                      help="Number of attention gaussians for convolution with one-hot ASCII text.")
  parser.add_argument("--numlstms", action="store", dest="num_lstms", type=int, default=1,
                      help="Number of LSTM's to use after the window layer.")
  parser.add_argument("--checkpointfile", action="store", dest="checkpoint_file", type=str, default=None,
                      help="Location of a checkpoint file to be loaded (for additional training or running).")

  args = parser.parse_args()

  data_dir = args.data_dir
  num_mix_components = args.num_mix_components
  num_att_components = args.num_att_components
  checkpoint_file = args.checkpoint_file
  num_lstms = args.num_lstms
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
  mdn_model = AttentionMDN(session, input_size, num_att_components, num_mix_components, 250, alphabet_size=dh.alphabet_size(), save=True, dropout=1.0, l2_penalty=1e-8)
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
  
  # Save off a pickle of the alphabet and one-hot encodings.
  dh.save_alphabet(save_dir)
  writer = tf.summary.FileWriter(save_dir, graph=session.graph)

  test_vals = dh.get_test_batch(2, 300)
  print("test_vals ascii:", test_vals["ascii"][0])
  for i in range(args.num_iterations):
    print("\niteration number: ", i)
    start_time = datetime.datetime.now()
    train = dh.get_train_batch(32, 300)
    batch_time = datetime.datetime.now() - start_time
    #if i > 5000:
    #  mdn_model.validate_batch(train["X"], train["onehot"], train["y"])
    things = mdn_model.train_batch(train["X"], train["onehot"], train["y"])

    if i % 100 == 0:
      print("  saving images", i)
      temp_test_vals = copy.deepcopy(test_vals)
      things = mdn_model.validate_batch(temp_test_vals["X"], temp_test_vals["onehot"], temp_test_vals["y"])
      save_weighted_deltas(things[1][0,:,:,:], things[3][0,:,:,], things[6][0,:], temp_test_vals["y"][0,:,:], save_dir, i, title=temp_test_vals["ascii"][0])
      save_mixture_weights(things[3][0,:,:], save_dir, suffix="prediction", offset=i, title=temp_test_vals["ascii"][0])
      save_attention_weights(things[-2][0,:,:], save_dir, suffix="window", ylabels="".join(dh.alphabet), title=temp_test_vals["ascii"][0], offset=i)
      save_attention_weights(things[-1][0,:,:], save_dir, suffix="window", ylabels="".join(test_vals["ascii"][0])  , title=temp_test_vals["ascii"][0], offset=i, filename="phi")

    if i % 500 == 0:
      mdn_model.save_params(os.path.join(save_dir, "mdn_model_ckpt"), i)

    #mdn_model.validate_batch(fake_train_in, fake_train_out)

    delta_batch = datetime.datetime.now() - start_time
    print("loss: ", things[0], "loss shape: ", things[0].shape)
    print("\t", "batch_time:", batch_time.total_seconds())
    print("\t", "delta_train:", delta_batch.total_seconds())
    with open(os.path.join(save_dir, "_error.dat"), "a") as f:
      f.write(str(i) + "," + str(things[0]) + ", " + str(delta_batch.total_seconds()) + "\n")
