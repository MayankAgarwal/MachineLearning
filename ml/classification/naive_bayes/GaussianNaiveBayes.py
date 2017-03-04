import numpy as np

class GaussianNaiveBayes(object):

	def __init__(self, class_priors=None):

		self.mean, self.variance, self.classes = None, None, None

		self.class_priors_provided = False
		self.class_priors = None

		if class_priors: 
			self.class_priors_provided = True
			self.class_priors = class_priors

	def __validateDataSize(self, x, y):
		"""
		Validates the size restrictions of X and Y
		Raises an exception if the restrictions are not followed. Returns None otherwise
		"""

		if type(x) != np.ndarray or type(y) != np.ndarray:
			raise TypeError('Invalid input data type')

		if x.shape[0] != y.shape[0]:
			raise AssertionError('No. of samples in feature data and output labels do not match')

		try:
			y.shape[1]
		except Exception as _:
			pass
		else:
			raise AssertionError('Shape of output label variable should be no. of samples x 1')

	def fit(self, x_train, y_train):
		"""
		Fit the Gaussian Naive Bayes on the training set

		Args:
			x_train (numpy array) : Training set of size no. of samples x no. of features
			y_train (numpy array) : Training set output labels of size no. of samples x 1

		Returns:
			None. The model is implicitly trained on calling this function.
		"""

		# Convert x_train and y_train to numpy arrays explicitly
		x_train = np.array(x_train)
		y_train = np.array(y_train)

		self.__validateDataSize(x_train, y_train)
		self.mean, self.variance, self.classes = self.__getMeansAndVariances(x_train, y_trains)
		self.__updateClassPriors(self, y_train)

	def __getMeansAndVariances(self, x_train, y_train):
		"""
		Compute the mean and variance for each class label

		Args:
			x_train (numpy array) : Input feature set data. Shape: No. of samples x no. of features
			y_train (numpy array) : Output class labels corresponding to training data points. Shape: no. of samples x 1

		Returns:
			mean (dict) : class label to the means of different features for that class
			variance (dict) : class label to the variance of different features for that class
			classes (numpy array) : a list of unique class labels in the training set
		"""

		classes = np.unique(y_train)
		means, variance = dict(), dict()

		for output_class in classes:

			class_mean = np.mean(x_train[y_train==output_class], axis=0)
			class_variance = np.var(x_train[y_train==output_class], axis=0)

			means[output_class] = class_mean
			variance[output_class] = class_variance

		return mean, variance, classes

	def __updateClassPriors(self, y_train):
		"""
		Updates the class prior probabilities

		Args:
			y_train (numpy array) : The output class label vector. Shape: no. of samples x 1

		Returns:
			None : Sets an instance variable with the class prior probabilities
		"""

		if self.class_priors_provided:
			return

		class_priors = {}
		total_instances = float(y_train.shape[0])

		for class_label in np.unique(y_train):
			instance_count = sum(y_train == class_label)
			class_prob = instance_count/total_instances
			class_priors[class_label] = class_prob

		self.class_priors = class_priors

	def __getGaussianProbability(self, feature_index, val, class_label):
		"""
		Computes the Gaussian probability of X = val given the mean and variance of the feature

		Args:
			feature_index (int) : The index of the feature in the original training set
			val (float) : Values of the input feature. X = val
			mean (float) : Mean of the feature from the training set
			class_label : Label of the output class

		Returns:
			The probability of a feature taking on a particular value given a class
		"""

		try:
			mean = self.mean[class_label][feature_index]
			variance = self.variance[class_label][feature_index]
		except KeyError as err:
			raise Exception('Class label not in training set')

		prefix = 1.0/np.sqrt(2 * np.pi * variance)
		postfix = np.exp( (-1.0 * (val - mean)**2) / (2*variance))

		return prefix * postfix

	def predict(self, x_test, get_probs=False):
		"""
		Returns the Naive Bayes classification of the X_test to one of the classes

		Args:
			x_test (numpy array) : Test feature set
			get_probs (optional) (boolean) : Set true if the class probabilites is required along with the classification.

		Returns:
			A list of following entities
			1. predicted class : The class label predicted by the classifier
			2. class probabilites (if get_probs = True) : Individual class probabilites for the feature set
		"""

		predicted_class = None
		predicted_class_prob = 0.0

		if get_probs:
			class_probabilities = {}

		for class_label in self.classes:

			class_prob = self.class_priors[class_label]

			for i in xrange(x_test.shape[0]):
				class_prob *= self.__getGaussianProbability(i, x_test[i], class_label)

			if get_probs: class_probabilities[class_label] = class_prob

			if class_prob > predicted_class_prob:
				predicted_class_prob = class_prob
				predicted_class = class_label

		result = [predicted_class]

		if get_probs: result.append(class_probabilities)

		return result

	def score(self, x_test, y_test):
		"""
		Returns the mean accuracy of classification on the given feature set
		
		Args:
			x_test (numpy array): Input feature set
			y_test (numpy array): Expected output label

		Returns:
			precision score (float) : Number of samples correctly classified by the classifer
		"""
		
		x_test = np.array(x_test)
		y_test = np.array(y_test)

		self.__validateDataSize(x_test, y_test)

		total_samples = float(y_test.shape[0])
		correctly_classified_count = 0.0

		for i in xrange(total_samples):
			correctly_classified_count += (self.predict(x_test[i,:]) == y_test[i])

		return correctly_classified_count/total_samples


