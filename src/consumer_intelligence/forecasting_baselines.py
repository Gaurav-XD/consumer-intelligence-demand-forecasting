"""Serializable baseline estimators for demand forecasting."""

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class SeasonalNaiveRegressor(RegressorMixin, BaseEstimator):
    """Predict demand from the same product's quantity twelve months earlier."""

    def fit(self, X, y=None):
        if "lag_12_quantity" not in X.columns:
            raise ValueError("Seasonal naive requires lag_12_quantity")
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.to_numpy()
        return self

    def predict(self, X):
        check_is_fitted(self, "feature_names_in_")
        if "lag_12_quantity" not in X.columns:
            raise ValueError("Seasonal naive requires lag_12_quantity")
        predictions = X["lag_12_quantity"]
        if predictions.isna().any():
            raise ValueError("Seasonal naive requires non-missing lag_12_quantity")
        return predictions.to_numpy(dtype=float)
