import numpy as np
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, model_type="gaussian"):
        self.model_type = model_type
        self.classes = None 
        self.priors = None
        self.conditional_probs = None

    def fit(self, X, y):
        """Train the Naïve Bayes classifier."""
        self.classes, class_counts = np.unique(y, return_counts=True)
        self.priors = {c: count / len(y) for c, count in zip(self.classes, class_counts)}
        self.conditional_probs = {}
        
        if self.model_type == "gaussian":
            self.conditional_probs = {c: {} for c in self.classes}
            for c in self.classes:
                X_c = X[y == c]
                self.conditional_probs[c]['mean'] = np.mean(X_c, axis=0)
                self.conditional_probs[c]['variance'] = np.var(X_c, axis=0) + 1e-9  # Smoothing, prevents the likelyhood of the result being zero
        
        elif self.model_type == "multinomial":#Calculates the features
            total_features = X.shape[1]
            self.conditional_probs = {c: {} for c in self.classes}
            for c in self.classes:
                X_c = X[y == c]
                feature_counts = np.sum(X_c, axis=0) + 1  # Laplace Smoothing, Also helps prevent that there are no zeros
                self.conditional_probs[c]['likelihoods'] = feature_counts / np.sum(feature_counts)
        else:
            raise ValueError("Unsupported model type")
    
    def _conditional_probability(self, x, class_probs):
        """Compute the conditional probability."""
        if self.model_type == "gaussian":
            mean, var = class_probs['mean'], class_probs['variance']
            return np.prod((1 / np.sqrt(2 * np.pi * var)) * np.exp(- (x - mean) ** 2 / (2 * var)))
        elif self.model_type == "multinomial":
            return np.prod(class_probs['likelihoods'] ** x)
        else:
            raise ValueError("Unsupported model type")
    
    def predict(self, X):
        """Predict the class labels for given input data."""
        predictions = []
        for x in X:
            predictors = {}
            for c in self.classes:
                prior = self.priors[c]
                likelihood = self._conditional_probability(x, self.conditional_probs[c]) #gives probability of that collumn
                predictors[c] = prior * likelihood
            predictions.append(max(predictors, key=predictors.get))
        return np.array(predictions)

# Example usage:
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split #will randomly generate the training data
    from sklearn.datasets import make_multilabel_classification 

    # This will generate the data set how we specify, 1000 data points, 5 features(colums), 
    X, y = make_multilabel_classification(n_samples = 10000, n_features = 5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y[:, 0], test_size=0.2, random_state=42)  # Use single label
    
    # Multinomial Accuracy Naïve Bayes
    nb_multinomial = NaiveBayesClassifier(model_type="multinomial")
    nb_multinomial.fit(X_train, y_train)
    y_pred = nb_multinomial.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Multinomial Naïve Bayes Accuracy: {accuracy:.2f}")
    
    #gaussian Accuacy
    nb_multinomial = NaiveBayesClassifier(model_type="gaussian")
    nb_multinomial.fit(X_train, y_train)
    y_pred = nb_multinomial.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Multinomial Naïve Bayes Accuracy: {accuracy:.2f}")
    