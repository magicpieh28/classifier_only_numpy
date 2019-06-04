import numpy as np
import pickle
import csv
from typing import List

from kadai1 import kadai1_dir
from kadai1.preprocessing import splits


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


def cal_accuracy(output: np.ndarray, target: np.ndarray, acc_list: List):
	acc = np.mean(np.equal(output, target)).item()
	if type(acc) == float:
		acc_list.append(acc)
	return acc


def cal_precision(output: np.ndarray, target: np.ndarray, pre_list: List):
	pre = (np.sum(target * output == 1) / np.sum(output)).item()
	if type(pre) == float:
		pre_list.append(pre)
	return pre


def cal_recall(output: np.ndarray, target: np.ndarray, rec_list: List):
	rec = (np.sum(target * output == 1) / np.sum(target)).item()
	if type(rec) == float:
		rec_list.append(rec)
	return rec


def cal_f1(precision: float, recall: float, f1_list: List):
	if precision != 0.0 or recall != 0.0:
		f1 = 2 * precision * recall / (precision + recall)
		if type(f1) == float:
			f1_list.append(f1)
		return f1


def eval(test_file: csv, batch_num: int = 5, feature_num: int = 3):
	model = pickle.load(open('kadai1.model', mode='rb'))

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

		output = label2matrix(predict_label, batch_num, feature_num)
		target = batch['target']

		cal_accuracy(output=output, target=target, acc_list=acc_score)
		precision = cal_precision(output=output, target=target, pre_list=pre_score)
		recall = cal_recall(output=output, target=target, rec_list=rec_score)
		cal_f1(precision=precision, recall=recall, f1_list=f1_score)

	return sum(acc_score) / len(acc_score), \
	       sum(pre_score) / len(pre_score), \
	       sum(rec_score) / len(rec_score), \
	       sum(f1_score) / len(f1_score)


if __name__ == '__main__':
    test_file = kadai1_dir / 'test.csv'
    acc, pre, rec, f1 = eval(test_file)
    print(f'acc => {acc}')
    print(f'pre => {pre}')
    print(f'rec => {rec}')
    print(f'f1 => {f1}')