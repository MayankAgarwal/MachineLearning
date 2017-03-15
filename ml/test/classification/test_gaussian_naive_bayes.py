import os
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.sys.path.append(path)

import unittest
from sklearn.naive_bayes import GaussianNB as sklearn_GaussianNB
from classification.naive_bayes.GaussianNaiveBayes import GaussianNaiveBayes as mod_GaussianNB
from sklearn.datasets import make_classification

class TestGaussianNB(unittest.TestCase):

	def test_prediction(self):

		mod_clf = mod_GaussianNB()
		sklearn_clf = sklearn_GaussianNB()
		tolerance = 1e-3
		x, y = make_classification(n_samples = 100000, n_classes=5, n_informative=4)
		train_samples_count = int(0.8*x.shape[0])

		x_train = x[0:train_samples_count, :]
		y_train = y[0:train_samples_count]
		x_test = x[train_samples_count:, :]
		y_test = y[train_samples_count:]

		mod_clf.fit(x_train, y_train)
		sklearn_clf.fit(x_train, y_train)

		mod_score = mod_clf.score(x_test, y_test)
		sklearn_score = sklearn_clf.score(x_test, y_test)

		print "Self implemented score: %f" % mod_score
		print "Sklearn score: %f" % sklearn_score

		assert(abs(mod_score - sklearn_score) <= tolerance)


if __name__ == "__main__":
	unittest.main()