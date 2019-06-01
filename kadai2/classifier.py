import numpy as np
import random


def softmax(z: np.ndarray):
	return np.exp(z) / np.sum(np.exp(z))


class Classifier(object):
	def __init__(self, X: np.ndarray):
		self.W = np.random.random((X.shape[1]))
		self.b = np.array(0.)

	def forward(self, X: np.ndarray):
		z = np.dot(X, self.w) + self.b
		return softmax(z)

	def backward(self, X: np.ndarray, output: np.ndarray, target: np.ndarray):
		n = X.shape[0]
		grad = -np.dot(X, target - output) / n
		bias = np.mean(target - output)
		return grad, bias