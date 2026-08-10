import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from typing import Callable, List, Tuple
from itertools import combinations


class SBS:
    """
    Sequential Backward Selection (SBS) for feature selection.

    SBS greedily removes features from the full feature set until the desired
    number of features `k_features` is reached. At each step, the feature whose
    removal leads to the best performance (according to a scoring function) is
    eliminated.

    Parameters
    ----------
    estimator : BaseEstimator
        A scikit-learn compatible estimator that implements `fit` and `predict`.
    k_features : int
        The target number of features to select.
    scoring : Callable[[List[float], List[float]], float], default=accuracy_score
        A scoring function with signature `scoring(y_true, y_pred) -> float`.
        Higher values indicate better performance.
    test_size : float, default=0.25
        Proportion of the data to use as the validation set for scoring.
    random_state : int, default=1
        Random seed for reproducible train/validation split.

    Attributes
    ----------
    indices_ : tuple
        The indices of the currently selected features (after fitting).
    subsets_ : list of tuples
        A list of feature index tuples at each step, starting with the full set
        and ending with the final selection.
    scores_ : list of float
        The validation scores corresponding to each subset in `subsets_`.
    k_score_ : float
        The validation score of the final subset (size = `k_features`).
    n_features_in_ : int
        The number of features in the input data.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        k_features: int,
        scoring: Callable[[List[float], List[float]], float] = accuracy_score,
        test_size: float = 0.25,
        random_state: int = 1,
    ) -> None:
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.scoring = scoring
        self.test_size = test_size
        self.random_state = random_state

        self.indices_ = None
        self.subsets_ = []
        self.scores_ = []
        self.k_score_ = None
        self.n_features_in_ = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> SBS:
        """
        Run the SBS algorithm to select the best `k_features` features.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : SBS
            The fitted instance.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        self.n_features_in_ = x.shape[1]

        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )

        self.indices_ = tuple(range(self.n_features_in_))
        self.subsets_ = [self.indices_]

        score = self._calc_score(x_train, x_val, y_train, y_val, self.indices_)
        self.scores_ = [score]

        dim = self.n_features_in_
        while dim > self.k_features:
            scores = []
            subsets = []

            for subset in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(x_train, x_val, y_train, y_val, subset)
                scores.append(score)
                subsets.append(subset)

            best_idx = np.argmax(scores)
            self.indices_ = subsets[best_idx]
            self.subsets_.append(self.indices_)
            self.scores_.append(scores[best_idx])
            dim -= 1

        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Reduce the feature set to the selected features.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        x_selected : np.ndarray of shape (n_samples, k_features)
            Data with only the selected features.
        """
        x = np.asarray(x)
        return x[:, self.indices_]

    def _calc_score(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        indices: Tuple[int, ...],
    ) -> float:
        """
        Compute the validation score for a given subset of features.

        Parameters
        ----------
        x_train, x_val, y_train, y_val : np.ndarray
            Training and validation data.
        indices : tuple of int
            Indices of the features to use.

        Returns
        -------
        score : float
            The computed scoring metric.
        """
        est = clone(self.estimator)
        est.fit(x_train[:, indices], y_train)
        y_pred = est.predict(x_val[:, indices])
        return self.scoring(y_val, y_pred)
