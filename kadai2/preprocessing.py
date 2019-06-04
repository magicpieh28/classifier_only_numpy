import csv
from typing import List
import numpy as np
import random

from kadai2 import kadai2_dir


def read_data(file: csv):  # データを読み込む。
    with file.open(mode='r') as f:
        lines = list(csv.reader(f))[1:]  # Headerは特徴量を含んでいない。
        features = [list(map(float, line[:2])) for line in lines]  # 特徴量が2つであるため。
        targets = [line[2] for line in lines]
        return random.Random(2).sample(features, len(features)), \
               random.Random(2).sample(targets, len(targets))  # クラスラベルを混ぜる。


def batch(items: List, i: int, batch_num: int):  # バッチ状態にする。
    if len(items) % batch_num != 0:
        if len(items) < i * batch_num + batch_num:
            return [items[i * batch_num:]]
        else:
            return [items[i * batch_num: i * batch_num + batch_num]]
    else:
        return [items[i * batch_num: i * batch_num + batch_num]]


def one_hot_encode(target: np.ndarray):  # ターゲットラベルがアルファベットであるため、マトリックスに変換。
    classes = {'A': 0, 'B': 1, 'C': 2}
    one_hot_target = np.zeros((target.shape[0], len(classes)), dtype=np.float)
    for i in range(len(target)):
        one_hot_target[i][classes[target.item(i)]] = 1.0
    return one_hot_target


def splits(file: csv, batch_num: int):  # データセットをbuildする。
    features, targets = read_data(file)
    assert len(features) == len(targets)

    batches = []
    target_list = []
    for i in range(len(features) // batch_num):
        feat_batch = np.stack(batch(features, i, batch_num)).squeeze()
        target_batch = np.stack(batch(targets, i, batch_num)).reshape((-1, 1))

        target_list.append(target_batch)
        batches.append({'feature': feat_batch,
                        'target': one_hot_encode(target_batch)})

    return batches, target_list


if __name__ == '__main__':
    train_data = kadai2_dir / 'train.csv'
    test_data = kadai2_dir / 'test.csv'

    batches = splits(train_data, 5)
    print(batches[0])
