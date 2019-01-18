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

  parser = argparse.ArgumentParser(description="The script used to run a mixture density network that has been trained on handwriting and conditioned on ASCII text.")

  parser.add_argument("--traindata", action="store", dest="data_dir", type=str, required=True,
                      help="Specify the location of a training data pickle file, including the filename (usually in \"./data_clean/some_folder/data.pkl\"). \
                      This pickle file only exists if you've run data_cleaner.py on the original dataset.")
  parser.add_argument("--checkpointfile", action="store", dest="checkpoint_file", type=str, default=None,
                      help="Location of a checkpoint file to be loaded (for additional training or running).")
  parser.add_argument("--string", action="store", dest="ascii_string", type=str, required=True, default="this is a test",
                      help="The ASCII string that you want the network to write out as handwriting.")
  parser.add_argument("--numsamples", action="store", dest="num_samples", type=int, required=False, default=1,
                      help="The number of sequences of handwriting to generate from the given ASCII string..")
  parser.add_argument("--bias", action="store", dest="bias", type=float, required=False, default=0,
                      help="How heavily to bias sampling. Low/high bias values lead to consistent/diverse styles of handwriting across samples.")
  parser.add_argument("--maxdots", action="store", dest="max_dots", type=int, required=False, default=1500,
                      help="The maximum number of dots that will be used to generate the handwriting sample.")
  parser.add_argument("--numlstms", action="store", dest="num_lstms", type=int, required=True,
                      help="The number of LSTM layers to include after the window layer.")

  args = parser.parse_args()

  data_dir = args.data_dir
  num_mix_components = 20
  num_att_components = 10
  checkpoint_file = args.checkpoint_file
  ascii_string = args.ascii_string
  num_samples = args.num_samples
  max_dots = args.max_dots
  num_lstms = args.num_lstms
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
  mdn_model = AttentionMDN(session, input_size, num_att_components, num_mix_components, 250, alphabet_size=dh.alphabet_size(), save=True, num_lstms=num_lstms)
    
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
    dots, strokes, alphabet_weights, phi = mdn_model.run_cyclically(np.zeros((1,1,3)), np.stack([dh.ascii_to_one_hot(args.ascii_string)], axis=0), max_dots, bias)
    print("alphabet_weights shape:", alphabet_weights.shape)
    save_dots(dots, strokes, save_dir, i, suffix=str(bias), save_text=True)
    save_attention_weights(alphabet_weights[0,:,:], save_dir, suffix="window", ylabels=dh.alphabet, title=args.ascii_string, offset=i)