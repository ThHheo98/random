from sklearn.base import TransformerMixin, BaseEstimator


class LowercaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [x.lower() for x in X]
