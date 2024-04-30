from sklearn.base import BaseEstimator, TransformerMixin
from config import config
import numpy as np
import pandas as pd

class MeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        self.mean_dict = {col: X[col].mean() for col in self.variables}
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            X[col] = X[col].fillna(self.mean_dict[col])
        return X

class ModeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        self.mode_dict = {col: X[col].mode()[0] for col in self.variables}
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            X[col] = X[col].fillna(self.mode_dict[col])
        return X

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables_to_drop = variables_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X.drop(columns=self.variables_to_drop, inplace=True)
        return X

class DomainProcessing(BaseEstimator, TransformerMixin):
    def __init__(self, variable_to_modify=None, variable_to_add=None):
        self.variable_to_modify = variable_to_modify
        self.variable_to_add = variable_to_add
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for feature in self.variable_to_modify:
            X[feature] += X[self.variable_to_add]
        return X

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        self.label_dict = {var: {k: i for i, k in enumerate(X[var].value_counts().sort_values(ascending=True).index, 0)} for var in self.variables}
        return self
    
    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.label_dict[feature])
        return X

class LogTransforms(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            X[col] = np.log(X[col] + 1)  # Using log1p to avoid issues with log(0)
        return X
