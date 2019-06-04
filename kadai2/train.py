import numpy as np
import csv
import pickle

from kadai2 import kadai2_dir
from kadai2.preprocessing import splits
from kadai2.classifier import SoftmaxRegression


def train(train_file: csv, batch_num: int = 5, epoch_num: int = 10,
          class_num: int = 3, feature_num: int = 2, lr: float = 0.01, l2: float = 1e-4):
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
        for i, batch in enumerate(batches):
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

    pickle.dump(model, open('kadai2.model', mode='wb'))
    return costs


if __name__ == '__main__':
    train_data = kadai2_dir / 'train.csv'
    test_data = kadai2_dir / 'test.csv'

    train(train_data)
