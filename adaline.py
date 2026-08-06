import numpy as np
from typing import List


class Adaline:
    """
    Adaline (Adaptive Linear Neuron) classifier.

    Implements a linear neuron trained with gradient descent on the mean squared
    error loss. The activation function is the identity, so the output is a real
    number, not a discrete class label.

    Attributes
    ----------
    eta : float
        Learning rate.
    iterations : int
        Number of passes over the training data.
    random_state : int
        Seed used for weight initialisation.
    _losses : List[float]
        History of the loss values after each epoch.
    sample_size : int
        Number of training samples.
    dimensions : int
        Number of features (including bias).
    _w : np.ndarray
        Weight vector (includes bias weight as the last element).
    _x : np.ndarray
        Training data with an added bias column (ones).
    _y : np.ndarray
        Target values.
    """

    def __init__(self, eta: float = 0.1, iterations: int = 10, random_state: int = 777) -> None:
        """
        Initialise the Adaline model.

        Parameters
        ----------
        eta : float, default=0.1
            Learning rate.
        iterations : int, default=10
            Number of training iterations.
        random_state : int, default=777
            Random seed for reproducible weight initialisation.
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
        Convert inputs to NumPy arrays, add a bias column, and initialise weights.

        The bias column is appended as a column of ones to the feature matrix.
        Weights are drawn from a normal distribution with mean 0 and std 0.01.

        Parameters
        ----------
        x : List[List[float]]
            Training feature matrix (samples × features).
        y : List[float]
            Target values.
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
        Compute the mean squared error loss.

        Returns
        -------
        float
            Mean squared error between predictions and true targets.
        """
        predictions = self._x @ self._w
        return np.mean((self._y - predictions) ** 2)

    def _gradient(self) -> np.ndarray:
        """
        Compute the gradient of the mean squared error loss w.r.t. weights.

        Returns
        -------
        np.ndarray
            Gradient vector of shape (dimensions,).
        """
        error = self._y - self._x @ self._w
        return (-2.0 / self.sample_size) * self._x.T @ error

    def fit(self, x: List[List[float]], y: List[float]) -> Adaline:
        """
        Train the Adaline model using batch gradient descent.

        Parameters
        ----------
        x : List[List[float]]
            Training feature matrix (samples × features).
        y : List[float]
            Target values.

        Returns
        -------
        Adaline
            The fitted model (self) for method chaining.
        """            
        self._preprocess(x, y)
        for _ in range(self.iterations):
            self._w -= self.eta * self._gradient()
            self._losses.append(self._loss())
        return self

    def _predict(self, x: np.ndarray) -> float:
        """
        Internal prediction for a single sample (already with bias included).

        Parameters
        ----------
        x : np.ndarray
            Feature vector including bias term (length = dimensions).

        Returns
        -------
        float
            Linear output (activation is identity).
        """
        return Adaline.activation(self._w @ x)

    def predict(self, x: List[float]) -> float:
        """
        Predict the output for a single sample.

        Parameters
        ----------
        x : List[float]
            Feature vector (without bias).

        Returns
        -------
        float
            Predicted value (real number).
        """
        x_with_bias = np.append(x, 1.0)
        return Adaline.activation(self._w @ x_with_bias)
    
    def losses(self) -> List[float]:
        """
        Return a copy of the loss history after each training epoch.

        Returns
        -------
        List[float]
            List of loss values.
        """
        return self._losses.copy()

    @staticmethod
    def activation(value: float) -> float:
        """
        Activation function (identity for Adaline).

        Parameters
        ----------
        value : float
            Input to the activation function.

        Returns
        -------
        float
            Same as input (identity).
        """
        return value
        
        
if __name__ == "__main__":
    p = Adaline(eta=0.0001, iterations=100000)
    X_train = [[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [1, 1]]
    y_train = [-1, -1, -1, 1, 1, 1]
    p.fit(X_train, y_train)
    print(p._w)
    print(p.predict([1, 1]))
    print(p.predict([0, 1]))
    print(p.predict([1, 0]))