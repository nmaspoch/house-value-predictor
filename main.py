import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

from sklearn.linear_model import LinearRegression

from sklearn import svm

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

def test_model(x, y, training_models):
    for training_model in training_models:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        model = training_model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(f"{type(training_model).__name__} model train score: {model.score(x_train, y_train)}")
        print(f"{type(training_model).__name__} model test score: {model.score(x_test, y_test)}")
        # plt.scatter(y_test, y_pred, c="green")
        # plt.xlabel("Price (in $1000s)")
        # plt.ylabel("Predicted value")
        # plt.title(f"True value vs predicted value : {type(training_model).__name__} model")
        # plt.show()

housing = fetch_california_housing()
df = pd.DataFrame(housing.data)
df.columns = housing.feature_names
df["price"] = housing.target

x = df.drop(["price"], axis=1)
y = df["price"]

training_models = [RandomForestRegressor(), LinearRegression(), XGBRegressor()]
test_model(x, y, training_models)