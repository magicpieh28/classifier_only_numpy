import numpy as np
import pickle
import csv
from typing import List

from kadai1 import kadai1_dir
from kadai1.preprocessing import splits


def num2label(predict_label: np.ndarray):
	"""
	モデルが予測したラベルの情報は、数値化されているため、ターゲットラベルのクラスと比較しやすく
	アルファベットに表し直すための関数。
	:param predict_label: モデルが予測したラベル情報。
	:return: 数値をアルファベットに変換し表現したリスト。
	"""
	labels = {0: 'A', 1: 'B', 2: 'C'}
	alpha_labels = [None] * len(predict_label)
	for i in range(len(predict_label)):
		alpha_labels[i] = labels[predict_label[i]]
	return alpha_labels


def label2matrix(predict_label: np.ndarray, batch_num: int, feature_num: int):
	"""
	Accuracy計算のため、モデルが予測し変換したラベル情報をマトリックスにするための関数。
	ミニバッチ中のターゲット値はマトリックスであるため、形をそろう必要がある。
	:param predict_label: モデルから予測されたラベル情報であり、ベクトルである。
	:param batch_num: ミニバッチサイズ。
	:param feature_num: 特徴量の数。
	:return: マトリックスに変換した予測ラベル情報。ミニバッチ中のターゲットマトリックスと同じshapeを持つ。
	"""
	matrix = np.zeros((batch_num, feature_num), dtype=np.float)
	for i in range(len(predict_label)):
		matrix[i][predict_label[i]] = 1.0
	return matrix


def cal_accuracy(output: np.ndarray, target: np.ndarray, acc_list: List):
	"""
	モデルの性能を測るための指標。
	:param output:　モデルが予測した値。
	:param target:　実際当てるべきラベル値。
	:param acc_list:　スコア情報を保管するリスト。
	:return:　Accuracy指標計算結果。
	"""
	acc = np.mean(np.equal(output, target)).item()
	if type(acc) == float:
		acc_list.append(acc)
	return acc


def cal_precision(output: np.ndarray, target: np.ndarray, pre_list: List):
	"""
	モデルが予測したラベルがターゲットラベルと一致した比率を計算する関数。
	:param output:　モデルが予測した値。
	:param target:　実際当てるべきラベル値。
	:param pre_list:　スコア情報を保管するリスト。
	:return:　Precision指標計算結果。
	"""
	pre = (np.sum(target * output == 1) / np.sum(output)).item()
	if type(pre) == float:
		pre_list.append(pre)
	return pre


def cal_recall(output: np.ndarray, target: np.ndarray, rec_list: List):
	"""
	ターゲットラベルの中で、実際モデルが当てたラベルの比率を計算する関数。
	:param output:　モデルが予測した値。
	:param target:　実際当てるべきラベル値。
	:param rec_list:　スコア情報を保管するリスト。
	:return:　Recall指標計算結果。
	"""
	rec = (np.sum(target * output == 1) / np.sum(target)).item()
	if type(rec) == float:
		rec_list.append(rec)
	return rec


def cal_f1(precision: float, recall: float, f1_list: List):
	"""
	Accuracyの場合、ラベル情報が不均衡である場合、モデルの性能をまともに測ることができない。
	これを補完するため、PrecisionとRecallの調和平均を求めることでモデルの性能を計り直す。
	:param precision:　Precision計算結果。
	:param recall:　Recall計算結果。
	:param f1_list:　スコア情報を保管するリスト。
	:return:　F1指標計算結果。
	"""
	if precision != 0.0 or recall != 0.0:  # nanを避けるため
		f1 = 2 * precision * recall / (precision + recall)
		if type(f1) == float:
			f1_list.append(f1)
		return f1


def eval(test_file: csv, batch_num: int = 5, feature_num: int = 3):
	"""
	テストファイルを用いて、モデルを検証するための関数。
	:param test_file: テストファイル。
	:param batch_num: ミニバッチサイズ。
	:param feature_num: 特徴量の数。
	:return: モデルの性能を測るためのAccuracy, Precision, Recall, F1指標の計算結果。
	"""
	model = pickle.load(open('kadai1.model', mode='rb'))  # 訓練モデルを呼び出す。

	test_data, targets = splits(test_file, batch_num)

	acc_score = []
	pre_score = []
	rec_score = []
	f1_score = []

	for i, batch in enumerate(test_data):
		predict_label = model.predict(batch['feature'])
		predict_str_label = num2label(predict_label)
		print(f'predict_str_label => {predict_str_label}')  # モデルが予測したラベルをアルファベットに表現したリスト。
		print(f'targets[i] => {targets[i]}')  # 実際のラベル情報をアルファベットで含んでいるマトリックス。

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
