import csv
from typing import List
import numpy as np
import random

from kadai2 import kadai2_dir


def read_data(file: csv):
	with file.open(mode='r') as f:
		lines = list(csv.reader(f))[1:]

		features = [line[:2] for line in lines]
		random.Random(4).shuffle(features)

		targets = [line[2] for line in lines]
		random.Random(4).shuffle(targets)

		return features, targets


def batch(items: List, i: int, batch_num: int):
	if len(items) % batch_num != 0:
		if len(items) < i * batch_num + batch_num:
			return [items[i * batch_num:]]
		else:
			return [items[i * batch_num: i * batch_num + batch_num]]
	else:
		return [items[i * batch_num: i * batch_num + batch_num]]


def splits(features: List[float], targets: List[str], batch_num: int):
	assert len(features) == len(targets)
	f_batchs = []
	t_batchs = []
	for i in range(len(features) // batch_num):
		f_batchs.append(np.stack(batch(features, i, batch_num)).squeeze())
		t_batchs.append(np.stack(batch(targets, i, batch_num)).reshape(-1, 1))
	return f_batchs, t_batchs


if __name__ == '__main__':
	train_data = kadai2_dir / 'train.csv'
	test_data = kadai2_dir / 'test.csv'

	features, targets = read_data(train_data)
	f_batchs, t_batchs = splits(features, targets, 5)
	print(f'f_batchs => {f_batchs}')
	print(f't_batchs => {t_batchs}')