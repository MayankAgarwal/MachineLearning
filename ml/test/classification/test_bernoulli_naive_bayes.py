import os
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.sys.path.append(path)

import unittest
from sklearn.naive_bayes import BernoulliNB as sklearn_BernoulliNB
from classification.naive_bayes.BernoulliNaiveBayes import BernoulliNaiveBayes as mod_BernoulliNB
import numpy as np

class TestBernoulliNB(unittest.TestCase):

	def test_prediction(self):

		mod_clf = mod_BernoulliNB()
		sklearn_clf = sklearn_BernoulliNB()
		tolerance = 1e-3

		x = np.random.randint(0, 2, size=(100000, 100))
		y = np.random.randint(1, 11, size=(100000))

		train_sample_count = int(0.8 * x.shape[0])

		x_train = x[0:train_sample_count, :]
		y_train = y[0:train_sample_count]

		x_test = x[train_sample_count:, :]
		y_test = y[train_sample_count:]

		mod_clf.fit(x_train, y_train)
		sklearn_clf.fit(x_train, y_train)

		mod_score = mod_clf.score(x_test, y_test)
		sklearn_score = sklearn_clf.score(x_test, y_test)

		print "Self implemented score: %f" % mod_score
		print "Sklearn score: %f" % sklearn_score

		assert(abs(mod_score - sklearn_score) <= tolerance)

if __name__ == "__main__":
	unittest.main()