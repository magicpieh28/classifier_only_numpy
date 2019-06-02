import numpy as np


class SoftmaxRegression(object):
	def __init__(self, lr: float, l2: float, batch_num: int, class_num: int):
		self.lr = lr
		self.l2 = l2
		self.batch_num = batch_num
		self.class_num = class_num

		self.W_ = np.random.random((self.batch_num, self.class_num))
		self.b_ = np.array([0.01, 0.1, 0.1])

	def predict_prob(self, X):
		z = self.net(X)
		prob = self.softmax(z)
		return prob

	def net(self, X: np.ndarray):
		return self.softmax(np.dot(X, self.W_) + self.b_)

	@staticmethod
	def softmax(z: np.ndarray):
		return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

	@staticmethod
	def find_argmax(z: np.ndarray):
		return z.argmax(axis=1)

	@staticmethod
	def cross_entropy(output: np.ndarray, target: np.ndarray):
		return - np.sum(target * np.log(output), axis=1)

	def calculate_cost(self, output: np.ndarray, target: np.ndarray):
		return np.mean(self.cross_entropy(output, target))
