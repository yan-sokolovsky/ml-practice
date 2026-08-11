import numpy as np
from typing import List


class LogisticRegression:
    """
    A simple binary logistic regression model trained using gradient descent.

    The model uses a sigmoid activation function and cross-entropy loss.
    It supports fitting on a dataset and making predictions for single samples.
    Training progress (loss per iteration) is stored and can be retrieved.
    """

    def __init__(self, eta: float = 0.1, iterations: int = 10, random_state: int = 777) -> None:
        """
        Initialize the logistic regression model.

        Parameters
        ----------
        eta : float, default=0.1
            Learning rate for gradient descent.
        iterations : int, default=10
            Number of training epochs.
        random_state : int, default=777
            Seed for random weight initialization to ensure reproducibility.
        """
        self.eta = eta
        self.iterations = iterations
        self.random_state = random_state

        self._losses: List[float] = []
        self.sample_size: int = 0
        self.dimensions: int = 0
        self._w: np.ndarray = None
        self._x: np.ndarray = None
        self._y: np.ndarray = None

    def _preprocess(self, x: List[List[float]], y: List[float]) -> None:
        """
        Convert inputs to NumPy arrays, add a bias column, and initialize weights.

        Parameters
        ----------
        x : List[List[float]]
            Training features (rows are samples, columns are features).
        y : List[float]
            Target labels (0 or 1).
        """
        self._x = np.array(x)
        self._x = np.hstack((self._x, np.ones((len(x), 1))))
        self._y = np.array(y)

        self.sample_size = self._x.shape[0]
        self.dimensions = self._x.shape[1]

        np.random.seed(self.random_state)
        self._w = np.random.normal(loc=0.0, scale=0.01, size=self.dimensions)

    def _loss(self) -> float:
        """
        Compute the binary cross-entropy loss.

        Returns
        -------
        float
            Mean cross-entropy loss over the training set.
        """
        predictions = self.activation(self._x @ self._w)
        eps = 1e-15
        predictions = np.clip(predictions, eps, 1 - eps)
        return -np.mean(self._y * np.log(predictions) + (1 - self._y) * np.log(1 - predictions))

    def _gradient(self) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to the weights.

        Returns
        -------
        np.ndarray
            Gradient vector of shape (dimensions,).
        """
        output = self.activation(self._x @ self._w)
        errors = self._y - output
        return self._x.T.dot(errors) / self.sample_size

    def fit(self, x: List[List[float]], y: List[float]) -> LogisticRegression:
        """
        Train the model on the given dataset.

        Parameters
        ----------
        x : List[List[float]]
            Training features.
        y : List[float]
            Training labels (0 or 1).

        Returns
        -------
        LogisticRegression
            The trained model instance (self).
        """
        self._preprocess(x, y)
        for _ in range(self.iterations):
            self._w += self.eta * self._gradient()
            self._losses.append(self._loss())
        return self

    def _predict(self, x: np.ndarray) -> float:
        """
        Predict the probability for a single sample (internal version, expects array with bias).

        Parameters
        ----------
        x : np.ndarray
            Input features including bias term.

        Returns
        -------
        float
            Predicted probability in [0, 1].
        """
        return self.activation(self._w @ x)

    def predict(self, x: List[float]) -> float:
        """
        Predict the probability for a single sample.

        Parameters
        ----------
        x : List[float]
            Feature vector for one sample (without bias).

        Returns
        -------
        float
            Predicted probability in [0, 1].
        """
        x_with_bias = np.append(x, 1.0)
        return self.activation(self._w @ x_with_bias)

    def losses(self) -> List[float]:
        """
        Return a copy of the loss history after training.

        Returns
        -------
        List[float]
            List of loss values per iteration.
        """
        return self._losses.copy()

    @staticmethod
    def activation(value: float) -> float:
        """
        Sigmoid activation function with clipping to avoid numerical overflow.

        Parameters
        ----------
        value : float
            Input to the sigmoid function.

        Returns
        -------
        float
            Sigmoid of the input, clamped to (0, 1) effectively.
        """
        return 1. / (1. + np.exp(-np.clip(value, -250, 250)))
        
        
if __name__ == "__main__":
    p = LogisticRegression(eta=0.001, iterations=100000)
    X_train = [[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [1, 1]]
    y_train = [0, 0, 0, 1, 1, 1]
    p.fit(X_train, y_train)
    print(p._w)
    print(p.predict([1, 1]))
    print(p.predict([0, 1]))
    print(p.predict([1, 0]))