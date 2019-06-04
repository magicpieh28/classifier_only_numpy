import numpy as np


class SoftmaxRegression(object):
    def __init__(self, batch_num: int, class_num: int, feature_num: int):
        """
        パラメタを初期化するため。
        :param batch_num: ミニバッチのサイズ。
        :param class_num: 分類すべきクラスの数。
        :param feature_num: 取り扱う特徴量の数（アウトプットサイズ）。
        """
        self.batch_num = batch_num
        self.class_num = class_num
        self.feature_num = feature_num

        self.W_ = np.random.random((self.feature_num, self.class_num))
        self.b_ = np.array([0.01, 0.1, 0.1])

    def net(self, X: np.ndarray):
        """
        入力値Xを線形活性関数に通し出力値を得るためのメソッド。
        :param X:　特徴量をstackしたマトリックスであり、（バッチサイズ, クラスサイズ）のshapeを持っている。
        :return:　Xに重みとバイアスを加え、活性関数Softmaxに通した出力値z。
        """
        return self.softmax(np.dot(X, self.W_) + self.b_)

    @staticmethod
    def softmax(z: np.ndarray):
        """
        バイナリー分類機のLogistic回帰モデルの一般化バージョンである、Softmax回帰モデルの活性関数。
        :param z: ネットワークの入力値Xに重みとバイアスを加えた値。
        :return: zを活性関数Softmaxに通した値であり、（-1, 1）の範囲内の確率分布を持つ。一行の値の和は1になる。
        """
        return (np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z)), axis=0)).T

    @staticmethod
    def find_argmax(prob: np.ndarray):
        """
        活性関数Softmaxをと教えられた確率から、最もずくすると思われるクラスを得るためのメソッド。
        :param prob: 活性関数Softmaxに通して得られた確率情報を持っているマトリックス。
        :return: 属する可能性が最も大きいクラス。
        """
        return prob.argmax(axis=1)

    @staticmethod
    def cross_entropy(output: np.ndarray, target: np.ndarray):
        """
        実際モデルが予測した結果と、ターゲットとの違いをエントロピーで測るためのメソッド。
        変換される値を最小化することでロスを減らすことができる。
        :param output: 活性関数Softmaxに通して得られた確率情報を含んでいるマトリックス。
        :param target: 実際当てて欲しいターゲットラベル。
        :return: ロスのエントロピーを計算した値。
        """
        return - np.sum(target * np.log(output), axis=0)

    def calculate_cost(self, cross_entropy, l2):
        """
        重みとバイアスの更新のため費用関数を計算する必要があり、そのためのメソッド。
        :param cross_entropy: ロスのエントロピーを計算した値。
        :param l2: 過学習を防ぐべく行う正規化のためのパラメタλ。
        :return:　重みとバイアスをアップデートする際に必要になる費用関数の計算結果。
        """
        l2_term = l2 * np.sum(self.W_ ** 2)
        return 0.5 * np.mean(cross_entropy + l2_term)

    def update(self, grad: np.ndarray, loss: np.ndarray, lr: float, l2: float):
        """
        重みとバイアスを更新するためのメソッド。
        :param grad:　グラディエント計算の値。
        :param loss:　予測したラベルとターゲットとの差を計算した値。
        :param lr:　学習率。
        :param l2:　正規化パラメタλ。
        :return:　更新されたモデルの重みとバイアス。
        """
        self.W_ -= (lr * grad + lr * l2 * self.W_)
        self.b_ -= (lr * np.sum(loss, axis=0))
        return self.W_, self.b_

    def predict(self, X: np.ndarray):
        """
        予測した確率に沿って、ラベルを選んで出力するためのメソッド。
        :param X: ネットワークの入力値。
        :return: 予測ラベルベクトル。
        """
        z = self.net(X)
        prob = self.softmax(z)
        return self.find_argmax(prob)
