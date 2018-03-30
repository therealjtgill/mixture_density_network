import os
import datetime
import sys
import shutil
import numpy as np
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


def xml_to_csv2(raw_data_location, clean_data_location, min_sequence_length=300, uid=""):

	if not os.path.isdir(clean_data_location):
		os.makedirs(clean_data_location)

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
								#f.write(point.attrib["x"], ",", point.attrib["y"], ",", "1")
							file_data.append(np.asarray([[0, 0, 0]]))
						#print("length of file_data:", len(file_data))
				all_data.append(np.vstack(file_data))
				print("file data shape:", all_data[-1].shape)

	all_data_ = np.vstack(all_data)
	print(all_data_.shape, all_data_[0:10, :])
	#all_data_[:, 0:2] -= np.mean(all_data_[:, 0:2], axis=0)
	mean = np.mean(all_data_[:, 0:2], axis=0)
	#all_data_[:, 0:2] /= np.std(all_data_[:, 0:2], axis=0)
	std = np.std(all_data_[:, 0:2], axis=0)

	#all_data_ -= mean
	#all_data_ /= std
	print("mean shape:", mean.shape)

	# Save all data as a single CSV file because YOLO.
	#csv_out = os.path.join(clean_data_location, "handwriting" + uid + ".csv")
	#np.savetxt(csv_out, all_data_, delimiter=",")

	# Save all data in separate CSV files
	for i in range(len(all_data)):
		if len(all_data[i]) >= min_sequence_length:
			csv_out = os.path.join(clean_data_location, "handwriting" + uid + str(i) + ".csv")
			#all_data[i][:, 0:2] = (all_data[i][:, 0:2] - mean)/std
			all_data[i][:, 0:2] = all_data[i][:, 0:2]/std
			np.savetxt(csv_out, all_data[i], delimiter=",")

if __name__ == "__main__":

	date_str = str(datetime.datetime.today()).replace(":", "-")
	#raw_data_location = \
	#	"C:\\Users\\gsaa\\Downloads\\lineStrokes-all.tar\\lineStrokes-all\\lineStrokes"
	raw_data_location = \
		"data_raw/original_with_transcriptions"
	clean_data_location = \
		"data_clean/data_" + date_str
	#copy_files(raw_data_location, out_data_location)

	xml_to_csv2(raw_data_location, clean_data_location, uid="full")