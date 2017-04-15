
import numpy as np
import collections

def file2string(filename):
  with open(filename, "r") as f:
		return f.read()

def string2dicts(data):
	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	chars, _ = list(zip(*count_pairs))
	char2id = dict(zip(chars, range(len(chars))))
	id2char = dict(zip(range(len(chars)), chars))
	return char2id, id2char

def string2ids(data, char2id):
	return [char2id[char] for char in data if char in char2id]

def read_data(data_path, file_names):
	train_file, valid_file, test_file = file_names
	train_path = data_path + train_file
	valid_path = data_path + valid_file
	test_path  = data_path + test_file
	train_string = file2string(train_path)
	valid_string = file2string(valid_path)
	test_string  = file2string(test_path)
	char2id, id2char = string2dicts(train_string)
	train_list = string2ids(train_string, char2id)
	valid_list = string2ids(valid_string, char2id)
	test_list  = string2ids(test_string, char2id)
	train_array = np.array(train_list, np.int32)
	valid_array = np.array(valid_list, np.int32)
	test_array = np.array(test_list, np.int32)
	t = len(train_array)
	tv = len(valid_array)
	te = len(test_array)
	train_data = train_array[0:t-1], train_array[1:t]
	valid_data = valid_array[0:tv-1], valid_array[1:tv]
	test_data = test_array[0:te-1], test_array[1:te]
	data = train_data, valid_data, test_data
	dicts = char2id, id2char
	v = len(char2id)
	return data, dicts, v

