import numpy as np
from typing import List, Union


class Perceptron:
    """
    A simple binary perceptron classifier.

    The perceptron learns a linear decision boundary using the
    stochastic gradient descent update rule. It expects binary labels
    {0, 1} and uses a step activation function.

    Parameters
    ----------
    eta : float, default=0.1
        Learning rate.
    epochs : int, default=10
        Number of passes over the training data.
    random_state : int, default=777
        Seed for the random number generator used to initialise weights.
    """

    def __init__(self, eta: float = 0.1, epochs: int = 10, random_state: int = 777) -> None:
        """
        Initialise the Perceptron model.

        Parameters
        ----------
        eta : float, default=0.1
            Learning rate.
        epochs : int, default=10
            Number of training epochs.
        random_state : int, default=777
            Random seed for reproducible weight initialisation.
        """
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state

        self._errors: List[int] = []
        self._errors_counter: int = 0
        self.sample_size: int = 0
        self.dimensions: int = 0
        self._w: np.ndarray = None
        self._x: np.ndarray = None
        self._y: np.ndarray = None

    def _preprocess(self, x: List[List[float]], y: List[float]) -> None:
        """
        Convert inputs to NumPy arrays, add a bias column, and initialise weights.

        The bias is added as a column of ones (appended to the feature matrix).
        Weights are drawn from a normal distribution with mean 0 and std 0.01.

        Parameters
        ----------
        x : List[List[float]]
            Training features, shape (n_samples, n_features).
        y : List[float]
            Training labels, binary {0, 1}.
        """
        self._x = np.array(x)
        self._x = np.hstack((self._x, np.ones((self._x.shape[0], 1))))

        self._y = np.array(y)

        self.sample_size = self._x.shape[0]
        self.dimensions = self._x.shape[1]

        np.random.seed(self.random_state)
        self._w = np.random.normal(loc=0.0, scale=0.01, size=self.dimensions)

    def _weights_delta(self, point: np.ndarray, group: int) -> np.ndarray:
        """
        Compute the weight update for a single training point.

        If the point is correctly classified, the update is a zero vector.
        Otherwise, the update is (true_label - predicted_label) * point.

        Parameters
        ----------
        point : np.ndarray
            Feature vector (including bias term).
        group : int
            True label (0 or 1).

        Returns
        -------
        np.ndarray
            Delta vector to be added to the weight vector.
        """
        prediction = self._predict(point)
        if group == prediction:
            return np.zeros(self.dimensions)
        self._errors_counter += 1
        return (group - prediction) * point

    def fit(self, x: List[List[float]], y: List[float]) -> 'Perceptron':
        """
        Train the perceptron on the given data.

        The training proceeds for a fixed number of epochs. For each epoch,
        the misclassification count is recorded.

        Parameters
        ----------
        x : List[List[float]]
            Training features, shape (n_samples, n_features).
        y : List[float]
            Training labels, binary {0, 1}.

        Returns
        -------
        Perceptron
            The fitted perceptron instance.
        """
        self._preprocess(x, y)
        for _ in range(self.epochs):
            self._errors_counter = 0
            for point, group in zip(self._x, self._y):
                self._w += self.eta * self._weights_delta(point, int(group))
            self._errors.append(self._errors_counter)
        return self

    def _predict(self, x: np.ndarray) -> int:
        """
        Internal prediction for a single sample (with bias included).

        Parameters
        ----------
        x : np.ndarray
            Feature vector (including bias term).

        Returns
        -------
        int
            Predicted class label {0, 1}.
        """
        return Perceptron.activation(self._w @ x)

    def predict(self, x: Union[List[float], np.ndarray]) -> int:
        """
        Predict the class of a single new sample.

        The bias term is automatically added inside this method.

        Parameters
        ----------
        x : Union[List[float], np.ndarray]
            Feature vector without bias (length = n_features).

        Returns
        -------
        int
            Predicted class label {0, 1}.
        """
        x_with_bias = np.append(x, 1.0)
        return Perceptron.activation(self._w @ x_with_bias)

    def errors(self) -> List[int]:
        """
        Return the list of misclassification counts per epoch.

        Returns
        -------
        List[int]
            A copy of the error history.
        """
        return self._errors.copy()

    @staticmethod
    def activation(value: float) -> int:
        """
        Step activation function.

        Returns 1 if the input is strictly positive, otherwise 0.

        Parameters
        ----------
        value : float
            The weighted sum (dot product).

        Returns
        -------
        int
            Binary output {0, 1}.
        """
        return 0 if value <= 0 else 1
    

if __name__ == "__main__":
    p = Perceptron(eta=0.1, epochs=10)
    X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [0, 0, 0, 1]
    p.fit(X_train, y_train)
    print(p._w)
    print(p.predict([1, 1]))
    print(p.predict([0, 1]))
    print(p.predict([1, 0]))
    print(p.errors())