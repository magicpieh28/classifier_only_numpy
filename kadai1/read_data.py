import csv
from kadai1 import kadai1_dir


def read_data(file: csv):
	data_dic = {
		'f1': [],
		'f2': [],
		'f3': [],
		'f4': [],
		'target': []
	}

	with file.open(mode='r') as f:
		lines = list(csv.reader(f))[1:]
		for line in lines:
			for i, val in enumerate(data_dic.values()):
				val.append(line[i])

	return data_dic


if __name__ == '__main__':
	train_data = kadai1_dir / 'train.csv'
	test_data = kadai1_dir / 'test.csv'

	data_dic = read_data(train_data)
	print(data_dic)
	for kew, val in data_dic.items():
		print(val.__len__())
