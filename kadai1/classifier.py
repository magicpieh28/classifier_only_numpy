import numpy as np


class SoftmaxRegression(object):
	def __init__(self, batch_num: int, class_num: int, feature_num: int):
		self.batch_num = batch_num
		self.class_num = class_num
		self.feature_num = feature_num

		self.W_ = np.random.random((self.feature_num, self.class_num))
		self.b_ = np.array([0.01, 0.1, 0.1])

	def net(self, X: np.ndarray):
		return self.softmax(np.dot(X, self.W_) + self.b_)

	@staticmethod
	def softmax(z: np.ndarray):
		return (np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z)), axis=0)).T

	@staticmethod
	def find_argmax(prob: np.ndarray):
		return prob.argmax(axis=1)

	@staticmethod
	def cross_entropy(output: np.ndarray, target: np.ndarray):
		return - np.sum(target * np.log(output), axis=0)

	def calculate_cost(self, cross_entropy, l2):
		l2_term = l2 * np.sum(self.W_ ** 2)
		return 0.5 * np.mean(cross_entropy + l2_term)

	def update(self, grad: np.ndarray, loss: np.ndarray, lr: float, l2: float):
		self.W_ -= (lr * grad + lr * l2 * self.W_)
		self.b_ -= (lr * np.sum(loss, axis=0))
		return self.W_, self.b_

	def predict(self, X: np.ndarray):
		z = self.net(X)
		prob = self.softmax(z)
		return self.find_argmax(prob)
