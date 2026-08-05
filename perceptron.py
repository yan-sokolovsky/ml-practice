import numpy as np
from typing import List

class Perceptron():
    """
    Classical perceptron for binary classification.
    
    Attributes:
        eta (float): Learning rate.
        epochs (int): Number of epochs.
        random_state (int): Seed for the random number generator (not currently used).
        _x (np.ndarray): Feature matrix with bias column.
        _y (np.ndarray): Target labels.
        _w (np.ndarray): Weights (including bias).
        _errors (List[int]): List of error counts by epoch.
    """
    def __init__(self, eta: float = 0.1, epochs: int = 10, random_state: int = 777) -> None:
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self._errors = []
        self._w = None
        self._x = None
        self._y = None
        
    def _preprocess(self, x: List[List[float]], y: List[float]) -> None:
        """
        Transforms the input data: adds a column of ones (bias) and stores the target values.
        
        Args:
            x: Feature matrix (without bias).
            y: Label vector (0 or 1).
        """
        self._errors = []
        self._x = np.array(x)
        self._y = np.array(y)
        self._x = np.hstack((self._x, np.ones((len(x), 1))))
        
    def fit(self, x: List[List[float]], y: List[float]) -> Perceptron:
        """
        Trains a perceptron on the data.
        
        Args:
            x: Feature matrix (without bias).
            y: Label vector (0 or 1).
            
        Returns:
            self: (for call chaining).
        """
        self._preprocess(x, y)
        np.random.seed(self.random_state)
        self._w = np.random.normal(loc=0, scale=0.01, size=self._x.shape[1])
        for _ in range(self.epochs):
            errors_counter = 0
            for point, group in zip(self._x, self._y):
                prediction = self._predict_with_bias(point)
                if group - prediction != 0:
                    errors_counter += 1
                    self._w += self.eta * (group - prediction) * point
            self._errors.append(errors_counter)
        return self

    def predict(self, x: List[float]) -> int:
        """
        A method for predicting the class of a single data point.
        
        Args:
            x: Feature vector (WITHOUT a bias – this will be added automatically).
        """
        x_with_bias = np.append(x, 1.0)
        return 0 if np.dot(self._w, x_with_bias) <= 0 else 1
    
    def _predict_with_bias(self, x: np.ndarray) -> int:
        """
        A method for predicting the class of a single data point.
        
        Args:
            x: Feature vector (WITH a bias).
        """
        return 0 if np.dot(self._w, x) <= 0 else 1
    
    def errors(self) -> List[int]:
        """
        Returns a list of the number of errors by epoch.
        
        Returns:
            _errors: list of the number of errors by epoch.
        """
        return self._errors.copy()
        
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