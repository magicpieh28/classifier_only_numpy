import numpy as np
import pickle
import csv

from kadai2 import kadai2_dir
from kadai2.preprocessing import splits


def num2label(predict_label: np.ndarray):
	labels = {0: 'A', 1: 'B', 2: 'C'}
	alpha_labels = [None] * len(predict_label)
	for i in range(len(predict_label)):
		alpha_labels[i] = labels[predict_label[i]]
	return alpha_labels


def label2matrix(predict_label: np.ndarray, batch_num: int, feature_num: int):
	matrix = np.zeros((batch_num, feature_num), dtype=np.float)
	for i in range(len(predict_label)):
		matrix[i][predict_label[i]] = 1.0
	return matrix


def eval(test_file: csv, batch_num: int = 5, feature_num: int = 3):
	model = pickle.load(open('kadai2.model', mode='rb'))

	test_data, targets = splits(test_file, batch_num)

	acc_score = []
	pre_score = []
	rec_score = []
	f1_score = []

	for i, batch in enumerate(test_data):
		predict_label = model.predict(batch['feature'])
		predict_str_label = num2label(predict_label)
		print(f'predict_str_label => {predict_str_label}')
		print(f'targets[i] => {targets[i]}')

		predict_label = label2matrix(predict_label, batch_num, feature_num)
		accuracy = np.mean(np.equal(batch['target'], predict_label))
		precision = np.sum(batch['target'] * predict_label == 1) / np.sum(predict_label)
		recall = np.sum(batch['target'] * predict_label == 1) / np.sum(batch['target'])
		f1 = 2 * precision * recall / (precision + recall)

		acc_score.append(accuracy)
		pre_score.append(precision)
		rec_score.append(recall)
		f1_score.append(f1)

	return sum(acc_score) / len(acc_score), \
	       sum(pre_score) / len(pre_score), \
	       sum(rec_score) / len(rec_score), \
	       sum(f1_score) / len(f1_score)


if __name__ == '__main__':
    test_file = kadai2_dir / 'test.csv'
    acc, pre, rec, f1 = eval(test_file)
    print(f'acc => {acc}')
    print(f'pre => {pre}')
    print(f'rec => {rec}')
    print(f'f1 => {f1}')