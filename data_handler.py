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
		Takes in a list of CSV files and an array indicating what portion of
		the files will be training, test, and validation sets. All files are
		treated as separate entities. When batches are generated, a sequence
		pulled from a single file constitutes one of the elements of the batch.
		'''

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
		#self.data_files = pkl_file
		#self.data_all = [np.loadtxt(csv, delimiter=",") for csv in self.data_files]
		self.data_all = pickle.load(open(pkl_file, 'rb'))
		self.max_sequence_length = max([d[1].shape[0] for d in self.data_all])

		self.num_lines = len(self.data_all)
		np.random.shuffle(self.data_all)
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
		print("Splitting finished.")


	def get_batch(self, batch_size, sequence_length, dataset):

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
		print("batches pulled from following line indices:", line_indices)
		batch_in = []
		batch_out = []
		batch_ascii = []
		for i in line_indices:
			#print('Sequence length of line stroke at index ', i, data[i][1].shape[0])
			start_index = np.random.randint(0, data[i][1].shape[0] - sequence_length)
			batch_in.append(data[i][1][start_index:sequence_length + start_index, :])
			batch_out.append(data[i][1][start_index + 1:sequence_length + start_index + 1, :])
			batch_ascii.append(data[i][0])

		batch_in = np.stack(batch_in, axis=0)
		batch_out = np.stack(batch_out, axis=0)
		print(batch_in.shape)
		print(batch_out.shape)

		batch_set = {"X":batch_in, "y":batch_out, "ascii":batch_ascii}

		return batch_set


	def get_train_batch(self, batch_size, sequence_length):

		return self.get_batch(batch_size, sequence_length, "train")


	def get_test_batch(self, batch_size, sequence_length):
		
		return self.get_batch(batch_size, sequence_length, "test")


	def get_validation_batch(self, batch_size, sequence_length):
		
		return self.get_batch(batch_size, sequence_length, "validate")


if __name__ == "__main__":

	data_dir = "data_clean/data_2018-03-25-16-15-56.863776"
	data_files = os.listdir(data_dir)
	data_files = [os.path.join(data_dir, d) for d in data_files if ".csv" in d]
	dh = data_handler(data_files[0:100], [.7, .15, .15])
	dh.get_train_batch(64, 100)
	dh.get_test_batch(63, 99)
	dh.get_validation_batch(62, 98)
