import numpy as np
import os


class multi_file_data_handler(object):

	def __init__(self, csv_files, splits):

		# splits = [%_train, %_validate, %_test]
		self.data_files = csv_files
		data_all = [np.loadtxt(csv, delimiter=",") for csv in csv_files]
		self.data_all = list(filter(lambda d: len(d) > 400, data_all))
		print("num removed:", len(data_all) - len(self.data_all))
		print("length of all data:", len(self.data_all))

		lengths = [len(d) for d in self.data_all]
		min_length = min(lengths)
		max_length = max(lengths)
		#print("lengths of strokes:", lengths)
		print("min sequence length:", min_length)
		print("max sequence length:", max_length)

		num_data_files = len(self.data_all)

		num_train = int(num_data_files*splits[0])
		num_validate = int(num_data_files*splits[1])
		num_test = num_data_files - num_train - num_validate

		# Note that these are arrays of individual sequences. We know that all
		# training data is zero-mean and unit norm.
		self.data_train = self.data_all[0:num_train]
		self.data_validate = self.data_all[num_train:num_train + num_validate]
		self.data_test = self.data_all[num_train + num_validate:]

	def get_batch(self, batch_size, sequence_length, dataset):
		'''
		A batch is composed of individual stroke sets
		'''

		if dataset == "train":
			data = self.data_train
		elif dataset == "test":
			data = self.data_test
		elif dataset == "validate":
			data = self.data_validate

		if batch_size > len(data):
			batch_size = len(data)


class data_handler(object):

	def __init__(self, csv_file, splits):

		self.data_file = csv_file
		self.data_all = np.loadtxt(self.data_file, delimiter=",")

		num_points = self.data_all.shape[0]

		# Just to be weird, split the data into P segments, then pull the
		# appropriate percentages of training and test data out of the P segments.
		# I'm letting P be the number of people that contributed writing samples.

		P = 221
		section_length = int(num_points/P)
		sections = [section_length for _ in range(P - 1)]

		data_sections = np.split(self.data_all, sections, axis=0)

		data_train = []
		data_test = []
		data_validate = []

		for i in range(P):
			section_length = data_sections[i].shape[0]
			sections = [
				int(section_length*splits[0]),
				int(section_length*splits[0]) + int(section_length*splits[1])
			]
			train, test, validate = np.split(data_sections[i], sections, axis=0)
			data_train.append(train)
			data_test.append(test)
			data_validate.append(validate)

		self.data_train = np.concatenate(data_train, axis=0)
		self.data_test = np.concatenate(data_test, axis=0)
		self.data_validate = np.concatenate(data_validate, axis=0)
		print("train data shape:", self.data_train.shape)
		print("test data shape:", self.data_test.shape)
		print("validation data shape:", self.data_validate.shape)


	def get_batch(self, batch_size, sequence_length, dataset):
		
		if dataset == "train":
			data = self.data_train
		elif dataset == "test":
			data = self.data_test
		elif dataset == "validate":
			data = self.data_validate

		low = 0
		high = data.shape[0] - sequence_length - 1
		start_indices = np.random.randint(low, high, batch_size)
		print("start indices:", start_indices)
		batch = []
		batch_out = []
		for i in range(batch_size):
			ind = start_indices[i]
			batch.append(data[ind:ind+sequence_length, :])
			batch_out.append(data[ind+1:ind+1+sequence_length, :])

		batch = np.stack(batch, axis=0)
		batch_out = np.stack(batch_out, axis=0)
		print(batch.shape)
		print(batch_out.shape)

		batch_set = {"X":batch, "y":batch_out}

		return batch_set


	def get_train_batch(self, batch_size, sequence_length):

		return self.get_batch(batch_size, sequence_length, "train")


	def get_test_batch(self, batch_size, sequence_length):
		
		return self.get_batch(batch_size, sequence_length, "test")


	def get_validation_batch(self, batch_size, sequence_length):
		
		return self.get_batch(batch_size, sequence_length, "validate")


if __name__ == "__main__":
	#dh = data_handler("data_clean/handwriting.csv", [.7, .15, .15])
	#dh.get_train_batch(64, 100)
	#dh.get_test_batch(63, 99)
	#dh.get_validation_batch(62, 98)
	files = [os.path.join("data_clean", f) for f in os.listdir("data_clean") if "handwriting.csv" not in f]
	print(len(files[1:5000]))
	dh = multi_file_data_handler(files, [.7, .15, .15])