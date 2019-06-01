import csv
from typing import List
import numpy as np
import random

from kadai2 import kadai2_dir


def read_data(file: csv):
	with file.open(mode='r') as f:
		lines = list(csv.reader(f))[1:]
		features = [line[:2] for line in lines]
		targets = [line[2] for line in lines]
		return random.Random(2).sample(features, len(features)), \
		       random.Random(2).sample(targets, len(targets))


def batch(items: List, i: int, batch_num: int):
	if len(items) % batch_num != 0:
		if len(items) < i * batch_num + batch_num:
			return [items[i * batch_num:]]
		else:
			return [items[i * batch_num: i * batch_num + batch_num]]
	else:
		return [items[i * batch_num: i * batch_num + batch_num]]


def one_hot_encode(target: np.ndarray):
	classes = {'A': 0, 'B': 1, 'C': 2}
	one_hot_target = np.zeros((target.shape[0], len(classes)), dtype=np.float)
	for i in range(len(target)):
		one_hot_target[i][classes[target.item(i)]] = 1
	return one_hot_target


def splits(features: List[float], targets: List[str], batch_num: int):
	assert len(features) == len(targets)
	f_batchs = []
	t_batchs = []
	for i in range(len(features) // batch_num):
		feat_batch = np.stack(batch(features, i, batch_num)).squeeze()
		f_batchs.append(feat_batch)

		target_batch = np.stack(batch(targets, i, batch_num)).reshape((-1, 1))
		t_batchs.append(one_hot_encode(target_batch))

	return f_batchs, t_batchs


if __name__ == '__main__':
	train_data = kadai2_dir / 'train.csv'
	test_data = kadai2_dir / 'test.csv'

	features, targets = read_data(train_data)
	f_batchs, t_batchs = splits(features, targets, 5)
	print(f'f_batchs => {f_batchs}')
	print(f't_batchs => {t_batchs}')