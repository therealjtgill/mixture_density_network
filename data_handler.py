import numpy as np
import os
import pickle
import sys


class data_handler(object):
  '''
  Takes in a list of files containing CSV data. Each file is loaded as a
  separate set of sequences to be fed into the recurrent MDN defined in
  mdn.py.
  '''

  def __init__(self, pkl_file, splits):
    '''
    Takes in a pkl file and an array indicating what portions of the pickle
    will be training, test, and validation sets or (logical) creates an
    alphabet and dictionary that maps characters to one-hot encodings.
    '''

    self.MIN_DATA_LENGTH = 100

    try:
      assert(isinstance(pkl_file, str))
    except AssertionError:
      sys.exit("The variable \"pkl_file\" is a " + str(type(pkl_file)) + " and not a str!")

    try:
      assert(sum(splits) == 1.0)
    except AssertionError:
      print("The \"splits\" variable needs three components that sum to one.")
      sys.exit("This was provided:" + str(splits))

    print("Loading data. Processing", len(pkl_file), "files.")
    self.data_all = pickle.load(open(pkl_file, 'rb'))
    self.max_sequence_length = max([d[1].shape[0] for d in self.data_all])
    self.max_ascii_length = max([len(d[0]) for d in self.data_all])

    self.num_lines = len(self.data_all)
    if self.num_lines < self.MIN_DATA_LENGTH:
      print("Not enough data! Only found", self.num_lines, "lines of data.")
      print("Need at least", self.MIN_DATA_LENGTH, "lines of data.")
      sys.exit(-1)

    np.random.shuffle(self.data_all)

    # Pad character sequences with spaces so that batches all have the same
    # number of one-hot encodings.
    for i in range(self.num_lines):
      ascii_data = self.data_all[i][0]
      num_pad_spaces = self.max_ascii_length - len(ascii_data) + 1
      #print(ascii_data)
      self.data_all[i][0] = ascii_data + " "*num_pad_spaces
    print("Loading finished.")

    print("Splitting data.")
    num_train = int(self.num_lines*splits[0])
    num_test = int(self.num_lines*splits[1])
    num_validate = self.num_lines - (num_train + num_test)
    print("  Number train files: ", num_train)
    print("  Number test files:", num_test)
    print("  Number validate files:", num_validate)

    self.data_train = self.data_all[:num_train]
    self.data_test = self.data_all[num_train:num_train + num_test]
    self.data_validate = self.data_all[num_train + num_test:]
    self.alphabet = sorted("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ` .!?")
    self.one_hot_alphabet_dict = {char:np.eye(len(self.alphabet), dtype=np.float32)[i] for i, char in enumerate(self.alphabet)}
    # "@" denotes an end of sequence; used to pad ascii one-hots.
    self.one_hot_alphabet_dict['@'] = np.zeros_like(self.one_hot_alphabet_dict['a'])
    print("alphabet: ", self.alphabet)
    print("Splitting finished.")


  def save_alphabet(self, save_dir):
    '''
    Saves the class's alphabet and one-hot alphabet dictionary off as a pickle
    file so that running the synthesis network won't require initializing the
    entire data handler class.
    '''

    pickle.dump([self.alphabet, self.one_hot_alphabet_dict], open(os.path.join(save_dir, "alphabet.pkl"), "wb"))


  def alphabet_size(self):
    '''
  	Returns the number of characters being used to represent ASCII labels of
  	the line stroke data.
  	'''

    return len(self.alphabet)


  def get_batch(self, batch_size, sequence_length, dataset, pad_char='@'):
    '''
    One-stop shop to get batches from any of the sets.
  	'''

    try:
      assert(sequence_length <= self.max_sequence_length)
    except AssertionError:
      print("Sequence length " + str(sequence_length) + " is larger than the allowed " + str(self.max_sequence_length))
      print("Setting sequence length to " + str(self.max_sequence_length))
      sequence_length = self.max_sequence_length
    
    if dataset == "train":
      data = self.data_train
    elif dataset == "test":
      data = self.data_test
    elif dataset == "validate":
      data = self.data_validate

    low = 0
    high = len(data)
    line_indices = np.random.randint(low, high, batch_size)
    #print("batches pulled from following line indices:", line_indices)
    batch_in = []
    batch_out = []
    batch_ascii = []
    # Need to make these all the same length
    batch_ascii_one_hot = []
    max_num_chars = 0
    for i in line_indices:
      #print('Sequence length of line stroke at index ', i, data[i][1].shape[0])
      # Count the number of points in the entire linestroke, divide number of
      # points by the number of characters in the line. Reduce the number of
      # characters based on the start index offset and the sequence length.
      # Need to pad ASCII sequences with something to make their lengths match.
      num_points_in_line = data[i][1].shape[0]
      num_chars_in_line = len(data[i][0])
      if num_chars_in_line > max_num_chars:
        max_num_chars = num_chars_in_line
      num_chars_per_point = num_chars_in_line/num_points_in_line
      start_index = 0
      char_offset = int(start_index*num_chars_per_point)
      num_chars = int(sequence_length*num_chars_per_point)
      #print("num_chars:", num_chars)
      batch_in.append(data[i][1][start_index:sequence_length + start_index, :])
      batch_out.append(data[i][1][start_index + 1:sequence_length + start_index + 1, :])
      ascii_data = data[i][0][char_offset:char_offset + num_chars]
      #print("complete ascii data:", data[i][0])
      #print("truncated ascii data:", ascii_data)
      batch_ascii.append(ascii_data.lstrip().rstrip()) # Remove leading and trailing spaces.

    for i in range(len(batch_ascii)):
      temp_ascii = batch_ascii[i]
      if len(batch_ascii[i]) < max_num_chars:
        temp_ascii += pad_char*(max_num_chars - len(temp_ascii))
      batch_ascii_one_hot.append(self.ascii_to_one_hot(temp_ascii))

    batch_in = np.stack(batch_in, axis=0)
    #batch_in[0:2] /= 10
    batch_out = np.stack(batch_out, axis=0)
    #batch_out[0:2] /= 10
    batch_ascii_one_hot = np.stack(batch_ascii_one_hot, axis=0)
    #print(batch_in.shape)
    #print(batch_out.shape)
    batch_set = {"X":batch_in, "y":batch_out, "ascii":batch_ascii, "onehot": batch_ascii_one_hot}

    return batch_set


  def ascii_to_one_hot(self, ascii_string):

    one_hots = []
    for char in ascii_string:
      if char in self.alphabet or char == '@':
        one_hots.append(self.one_hot_alphabet_dict[char])
      else:
        one_hots.append(self.one_hot_alphabet_dict['`'])
    #return np.stack([self.one_hot_alphabet_dict[char.lower()] for char in ascii_string], axis=1).T #ugly
    return np.stack(one_hots)


  def get_train_batch(self, batch_size, sequence_length):

    return self.get_batch(batch_size, sequence_length, "train", '@')


  def get_test_batch(self, batch_size, sequence_length):
    
    return self.get_batch(batch_size, sequence_length, "test", '@')


  def get_validation_batch(self, batch_size, sequence_length):
    
    return self.get_batch(batch_size, sequence_length, "validate", '@')


if __name__ == "__main__":

#  data_dir = "data_clean/data_2018-03-25-16-15-56.863776"
#  data_files = os.listdir(data_dir)
#  data_files = [os.path.join(data_dir, d) for d in data_files if ".csv" in d]
  dh = data_handler("./data_clean/cleanedlabeleddata.pkl", [.7, .15, .15])
  dh.get_train_batch(64, 300)
  dh.get_test_batch(63, 299)
  dh.get_validation_batch(62, 298)
  print("Num training lines:", len(dh.data_train))
  print("Length of alphabet:", len(self.alphabet))
