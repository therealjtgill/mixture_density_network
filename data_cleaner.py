import argparse
import datetime
import numpy as np
import os
import pickle
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
						file_data.append(np.asarray([[x, y, 0.0]]))
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
				#clean_filename = file.split(".xml")[0] + ".csv"
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
									file_data.append(np.asarray([[0., 0., 0.0]]))
								else:
									file_data.append(np.asarray([[x - x_prev, y - y_prev, 0.0]]))
								x_prev = x
								y_prev = y
							# Mark last item in the list of points as the end of stroke.
							file_data[-1][0,-1] = 1.0
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
			# Just scale the data by its standard deviation values.
			all_data[i][:, 0:2] = all_data[i][:, 0:2]/std
			np.savetxt(csv_out, all_data[i], delimiter=",")


def extract_stroke_from_xml(file_location):
	'''
  Extracts strokesets from XML files.
	'''

	tree = ET.parse(file_location)
	root = tree.getroot()

	stroke_data = []
	ascii_data = []
	for strokeset in root:
		x_prev = 0.
		y_prev = 0.
		if strokeset.tag == "StrokeSet":
			for stroke in strokeset:
				for point in stroke:
					x = float(point.attrib["x"])
					y = float(point.attrib["y"])
					if len(stroke_data) == 0:
						stroke_data.append(np.asarray([[0., 0., 1.0]]))
					else:
						stroke_data.append(np.asarray([[x - x_prev, y - y_prev, 0.0]]))
					x_prev = x
					y_prev = y
				# Mark last item in the list of points as the end of stroke.
				stroke_data[-1][0,-1] = 1.0
			#print("length of stroke_data:", len(stroke_data))

	stroke_data = np.vstack(stroke_data)
	#print(stroke_data.shape, stroke_data[0:10, :])
	return stroke_data


def extract_ascii_and_stroke_from_xml(stroke_files, ascii_file):
  '''
  Line stroke file names have the pattern
    k10-103z-01.xml
  Ascii file names have the pattern
    k10-103z.txt
  The last number in the stroke file name indicates the line number of the
  ASCII CSR that contains the line of text corresponding to the strokeset.
  '''

  ascii_stroke_data = []
  
  print(ascii_file)
  with open(ascii_file, 'r') as f:
    lines = f.readlines()
    for i in range(len(lines)):
      if "CSR" in lines[i]:
        start_index = i + 2
        break
    if len(lines) - start_index != len(stroke_files):
      print("  Number of CSR lines in ascii file doesn't match number of stroke files... Bail out!")
      return ascii_stroke_data
    for i in reversed(range(len(stroke_files))):
      ascii_line_number = int(stroke_files[i][-6:-4])
      stroke_data = extract_stroke_from_xml(stroke_files[i])
      ascii_data = lines[len(lines) - 1 - len(stroke_files) + ascii_line_number].strip()
      #print('ascii_data:', ascii_data)
      #print('  line number:', ascii_line_number)
      #print('  line index being used:', len(lines) - len(stroke_files) - 1 + ascii_line_number)
      #print('  len(lines):', len(lines))
      #print('  len(stroke_files):', len(stroke_files))
      ascii_stroke_data.append([ascii_data, stroke_data])

  return ascii_stroke_data


def correlate_ascii_and_stroke_files(stroke_file_locations, ascii_file_locations, min_sequence_length=300):
  '''
  Grabs stroke and ascii files with similar names and groups them together,
  then extracts the information from them at the same time and groups that
  information together.
  '''

  ascii_stroke_data = []
  for i in range(len(ascii_file_locations)):
    ascii_filename = ascii_file_locations[i].split(os.sep)[-1].split('.')[0]
    ascii_file = ascii_file_locations[i]
    stroke_files = []
    #print('looking for stroke files with name: ', ascii_filename)
    for j in range(len(stroke_file_locations)):
      stroke_filename = stroke_file_locations[j].split(os.sep)[-1][:-7]
      #print('stroke fname:', stroke_filename, 'ascii fname:', ascii_filename, 'stroke fname before chop:', stroke_file_locations[j].split(os.sep)[-1], 'ascii fname before chop:', ascii_file_locations[i].split(os.sep)[-1])
      if ascii_filename == stroke_filename:
        #print('  found correlation')
        stroke_files.append(stroke_file_locations[j])
    #print('found files: ', stroke_files)
    if len(stroke_files) > 0:
      ascii_stroke_data += extract_ascii_and_stroke_from_xml(stroke_files, ascii_file)
  
  filtered_ascii_stroke_data = [d for d in ascii_stroke_data if d[1].shape[0] > min_sequence_length]

  return filtered_ascii_stroke_data


def get_training_files(raw_data_location, ext):
  
  file_list = []
  for root, dirs, files in os.walk(raw_data_location):
    #print("files:", files)
    for file in files:
      if file.endswith(ext):
        file_list.append(os.path.join(root, file))

  return file_list


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Takes XML data from the IAM online handwriting database and cleans it. \
                                                The only file structure that is supported is the lineStrokes data, \"\
                                                which can be downloaded from the IAM online website.")

  parser.add_argument("--rawdata", action="store", type=str, required=True,
                      help="The location of the raw (extracted) lineStroke XML files from the IAM online handwriting db.")
  parser.add_argument("--cleandata", action="store", type=str, required=True,
                      help="Where the clean data that has been extracted from XML files should be stored.")
  parser.add_argument("--asciidata", action="store", type=str,
                      help="Indicates that the data being loaded is lineStroke data, and that transcripts of each line should also be extracted.")

  args = parser.parse_args()
  clean_data_location = args.cleandata
  raw_data_location = args.rawdata
  ascii_data_location = args.asciidata

  date_str = str(datetime.datetime.today()).replace(":", "-")
  #copy_files(raw_data_location, out_data_location)

  #xml_with_annotations_to_csv(raw_data_location, clean_data_location, uid="full")

  if not os.path.isdir(clean_data_location):
    os.makedirs(clean_data_location)

  stroke_files = get_training_files(raw_data_location, ".xml")
  ascii_files = get_training_files(ascii_data_location, ".txt")

  #stroke = extract_stroke_from_xml(training_files[0])
  #np.savetxt('test.csv', stroke, delimiter=',')

  # Saves a list of tuples of the form [(ascii text, numpy matrix of strokes),]
  ascii_stroke_data = correlate_ascii_and_stroke_files(stroke_files, ascii_files)
  pickle.dump(ascii_stroke_data, open(os.path.join(clean_data_location, "cleanedlabeleddata.pkl"), "wb"))
