import numpy as np

class BernoulliNaiveBayes(object):

	def __init__(self, class_priors=None):

		self.class_prior_provided = False
		self.class_priors = None
		self._likelihood = {}

		if class_priors:
			self.class_prior_provided = True
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
		Fit the Binomial Naive Bayes on the training set

		Args:
			x_train (numpy array) : Training set of size no. of samples x no. of features
			y_train (numpy array) : Training output labels of size no. of samples x 1

		Returns:
			None. The model is implicitly trained on function call
		"""

		# Convert x_train and y_train to numpy arrays
		x_train = np.array(x_train)
		y_train = np.array(y_train)

		self.__validateDataSize(x_train, y_train)
		self.__computeClassPriors(y_train)
		self.__computeWordLikelihood(x_train, y_train)

	def __computeClassPriors(self, output_labels):
		"""
		Computes the priors for each class in the training set.

		Args:
			output_labels (numpy array) : The output class labels vector. Size: no. of samples x 1

		Returns:
			None. Updates the instance variable with the class priors
		"""

		if self.class_prior_provided:
			return

		class_priors = {}
		total_instances = float(output_labels.shape[0])

		for class_label in np.unique(output_labels):
			instance_count = sum(output_labels==class_label)
			prior = instance_count/total_instances
			class_priors[class_label] = prior

		self.class_priors = class_priors

	def __computeWordLikelihood(self, x_train, y_train):
		"""
		Computes the P(x|C) for every class and x - Likelihood of word occuring in a class

		Args:
			x_train (numpy array) : Input training data of size: No. of samples x no. of features
			y_train (numpy array) : Training output labels of size: No. of samples x 1

		Returns:
			None. Updates an instance variable with likelihoods of each word
		"""

		feature_count = x_train.shape[1]

		for class_label in np.unique(y_train):

			total_samples = sum(y_train == class_label)

			for feature_index in xrange(feature_count):
				word_count = sum(x_train[y_train == class_label, feature_index])
				word_likehood = float(word_count) / total_samples

				self._likelihood.setdefault(class_label, dict())[feature_index] = word_likehood

	def predict(self, x_test, get_probs=False):
		"""
		Returns the predicted class label for the given test instance

		Args:
			x_test (numpy array) : Test feature set
			get_probs (optional) (boolean) : Set true if the class probabilities is required along with the classification
		"""

		predicted_class, predicted_class_prob = None, 0.0
		class_probs = dict()
		prob_normalizer = 0.0

		for class_label in self.class_priors.keys():
			prior = self.class_priors[class_label]
			prob = prior

			for feature_index, probs in self._likelihood[class_label].iteritems():
				does_word_occur = x_test[feature_index]
				posterior = (probs*does_word_occur) + ( (1-probs)*(1-does_word_occur) )
				prob = prob * posterior

			if prob > predicted_class_prob:
				predicted_class_prob = prob
				predicted_class = class_label

			prob_normalizer += prob

		class_probs = self.__normalize_probabilities(class_probs, prob_normalizer)

		result = [predicted_class]

		if get_probs: result.append(class_probs)

		return result

	def __normalize_probabilities(self, probabilites, normalizer):
		"""
		Normalizes the probabilites computed by the classifier

		Args:
			probabilities (dict) : Probabilites of classes represented as key-value pairs
			normalizer (float) : The normalizing factor

		Returns:
			probabilities (dict) : Normalized probabilities
		"""

		for x, prob in probabilites.iteritems():
			probabilites[x] = float(prob)/normalizer

		return probabilites

	def score(self, x_test, y_test):
		"""
		Returns the mean accuracy of classification on the given data

		Args:
			x_test (numpy array) : Input feature set
			y_test (numpy array) : Expected output label

		Returns:
			precision score (float) : Percentage of samples correctly classified by the classifier
		"""

		x_test = np.array(x_test)
		y_test = np.array(y_test)

		self.__validateDataSize(x_test, y_test)

		total_samples = y_test.shape[0]
		correctly_classified_samples = 0.0

		for i in xrange(total_samples):
			correctly_classified_samples += (self.predict(x_test[i,:]) == y_test[i])

		return float(correctly_classified_samples) / total_samples
