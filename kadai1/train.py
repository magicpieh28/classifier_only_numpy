import numpy as np
import csv
import pickle

from kadai1 import kadai1_dir
from kadai1.preprocessing import splits
from kadai1.classifier import SoftmaxRegression


def train(train_file: csv, batch_num: int = 5, epoch_num: int = 10,
          class_num: int = 3, feature_num: int = 4, lr: float = 0.01, l2: float = 1e-4):
    """
    モデルを訓練するためのメソッド。
    訓練したモデルはPickleでセーブする。
    :param train_file: 訓練対象となるファイル。
    :param batch_num: ミニバッチサイズ。
    :param epoch_num: 訓練回数。
    :param class_num: ぶんるしたいクラスの数。（アウトプットサイズ）
    :param feature_num: 用いられる特徴量の数。
    :param lr: 学習率。
    :param l2: か学習を防ぐべく行う正規化のためのパラメタλ。
    :return: 予測ラベルとターゲットラベルとのさを計算した値。
    """
    print('build datum.')
    batches, targets = splits(file=train_file, batch_num=batch_num)

    print('model defined.')
    model = SoftmaxRegression(batch_num=batch_num,
                              class_num=class_num,
                              feature_num=feature_num)

    costs = []
    for i in range(epoch_num):
        for batch in batches:
            X = batch['feature']
            z = model.net(X)
            prob = model.softmax(z)

            loss = prob - batch['target']
            grad = np.dot(X.T, loss)

            model.W_, model.b_ = model.update(grad=grad, loss=loss, lr=lr, l2=l2)

            cross_ent = model.cross_entropy(output=prob, target=batch['target'])
            cost = model.calculate_cost(cross_entropy=cross_ent, l2=l2)
            costs.append(cost)

        print('cost: ')
        print(np.sum(costs) / len(costs))

    pickle.dump(model, open('kadai1.model', mode='wb'))
    print('model saved.')
    return costs


if __name__ == '__main__':
    train_data = kadai1_dir / 'train.csv'
    test_data = kadai1_dir / 'test.csv'

    train(train_data)