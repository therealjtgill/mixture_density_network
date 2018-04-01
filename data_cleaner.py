import argparse
import datetime
import numpy as np
import os
import shutil
import sys
import xml.etree.ElementTree as ET
np.set_printoptions(threshold=np.nan)


def copy_files(raw_data_location, out_data_location):

	for root, dirs, files in os.walk(raw_data_location):
		for file in files:
			if file.endswith(".xml"):
				#print(root, " things ", file)
				container = root.split(os.sep)[-2]
				#new_dir = os.path.join(out_data_location, container)
				#if not os.path.exists(new_dir):
				#	os.makedirs(new_dir)
				src = os.path.join(root, file)
				dst = os.path.join(out_data_location, file)
				shutil.copy(src, dst)


def xml_to_csv(raw_data_location, clean_data_location, uid=""):
	'''
	This is a legacy function for extracting XML data on stroke positions and
	pen up/down from XML files that do not contain character annotations.
	DO NOT USE.
	'''

	if not os.path.isdir(clean_data_location):
		os.makedirs(clean_data_location)

	all_data = []
	for root, dirs, files in os.walk(raw_data_location):
		#print("files:", files)
		for file in files:
			print("filename:", file)
			if file.endswith(".xml"):
				clean_filename = file.split(".xml")[0] + ".csv"
				tree = ET.parse(os.path.join(raw_data_location, file))
				root = tree.getroot()

				file_data = []
				for stroke in root[1]:
					file_data.append(np.asarray([[0, 0, 0]]))
					for point in stroke:
						x = float(point.attrib["x"])
						y = float(point.attrib["y"])
						file_data.append(np.asarray([[x, y, 1.0]]))
						#f.write(point.attrib["x"], ",", point.attrib["y"], ",", "1")
				#print("length of file_data:", len(file_data))
				all_data.append(np.vstack(file_data))

	all_data_ = np.vstack(all_data)
	print(all_data_.shape)
	all_data_[:, 0:2] -= np.mean(all_data_[:, 0:2], axis=0)
	all_data_[:, 0:2] /= np.std(all_data_[:, 0:2], axis=0)

	# Save all data as a single CSV file because YOLO.
	csv_out = os.path.join(clean_data_location, "handwriting" + uid + ".csv")
	np.savetxt(csv_out, all_data, delimiter=",")

	# Save all data in separate CSV files
	#for i in range(len(all_data)):
	#	csv_out = os.path.join(clean_data_location, "handwriting" + str(i) + ".csv")
	#	np.savetxt(csv_out, all_data[i], delimiter=",")


def xml_with_annotations_to_csv(raw_data_location, clean_data_location, min_sequence_length=300, uid=""):
	'''
	This function is tailored to extracting data from the IAM Online Handwriting
	database, where XML files contain both sets of strokes and transcriptions
	of what was written.

	This function is broken into two parts:
	  1. The stroke position and value extractor
	  2. The stroke information saver

	The stroke position and value extractor extracts (x,y) positions from the
	XML files, and saves in RAM the value (x_(i+1), y_(i+1)) - (x_i, y_i) as a
	vector.

	The data extracted from separate XML files are kept in separate arrays, and
	are later saved in separate CSV files. This is meant to make the selection of
	training batches more consistent.
	'''

	if not os.path.isdir(clean_data_location):
		os.makedirs(clean_data_location)

	# "all_data" will contain an array of rank-2 tensors. Elements of the
	# array will be the sets of strokes that belong to individual files in
	# the training set.
	all_data = []
	for root, dirs, files in os.walk(raw_data_location):
		#print("files:", files)
		for file in files:
			if file.endswith(".xml"):
				clean_filename = file.split(".xml")[0] + ".csv"
				tree = ET.parse(os.path.join(root, file))
				root = tree.getroot()

				file_data = []
				for strokeset in root:
					x_prev = 0.
					y_prev = 0.
					if strokeset.tag == "StrokeSet":
						for stroke in strokeset:
							for point in stroke:
								x = float(point.attrib["x"])
								y = float(point.attrib["y"])
								if len(file_data) == 0:
									file_data.append(np.asarray([[0., 0., 1.0]]))
								else:
									file_data.append(np.asarray([[x - x_prev, y - y_prev, 1.0]]))
								x_prev = x
								y_prev = y
							# Mark last item in the list of points as the end of stroke.
							file_data[-1][0,-1] = 0.0
						#print("length of file_data:", len(file_data))
				all_data.append(np.vstack(file_data))
				print("file data shape:", all_data[-1].shape)

	all_data_ = np.vstack(all_data)
	print(all_data_.shape, all_data_[0:10, :])
	mean = np.mean(all_data_[:, 0:2], axis=0)
	std = np.std(all_data_[:, 0:2], axis=0)
	print("mean shape:", mean.shape)

	# Save all data in separate CSV files.
	for i in range(len(all_data)):
		if len(all_data[i]) >= min_sequence_length:
			csv_out = os.path.join(clean_data_location, "handwriting" + uid + str(i) + ".csv")
			# Just scale the data by its standard deviation values; translation will
			# screw up the network's ability to predict.
			all_data[i][:, 0:2] = all_data[i][:, 0:2]/std
			np.savetxt(csv_out, all_data[i], delimiter=",")

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Takes XML data from the IAM online handwriting database and cleans it. \
																								Currently the only file structure that is supported is the \"original-xml-part.tar.gz \"\
																								file, which can be downloaded from the IAM online website.")

	parser.add_argument("--rawdata", action="store", dest="raw_data_location", type=str, required=True,
											description="The location of the raw (extracted) XML files from the IAM online handwriting db.")
	parser.add_argument("--cleandata", action="store", dest="clean_data_location", type=str,
											description="The ")

	date_str = str(datetime.datetime.today()).replace(":", "-")
	#raw_data_location = \
	#	"C:\\Users\\gsaa\\Downloads\\lineStrokes-all.tar\\lineStrokes-all\\lineStrokes"
	raw_data_location = \
		"data_raw/original_with_transcriptions"
	clean_data_location = \
		"data_clean/data_" + date_str
	#copy_files(raw_data_location, out_data_location)

	xml_with_annotations_to_csv(raw_data_location, clean_data_location, uid="full")