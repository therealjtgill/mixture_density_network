import numpy as numpy
import os
import data_handler as dh

if __name__ == "__main__":
	data_dir = "data_clean/data_2018-03-25-16-15-56.863776"
	data_files = os.listdir(data_dir)
	data_files = [os.path.join(data_dir, d) for d in data_files if ".csv" in d]

	dh = dh.data_handler(data_files, [.7, .15, .15])

	for i in range(100000000):
	    train = dh.get_train_batch(32, 300)