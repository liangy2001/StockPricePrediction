from old_files.new_lr import linear_regression_model

# from sklearn.datasets import load_boston
#
# boston = load_boston()
#
# bost = pd.DataFrame(boston['data'])
# bost.columns = boston['feature_names']
# bost.head()
#
# X = ((bost - bost.mean())/(bost.max() - bost.min()))
# y = boston['target']

import numpy as np
import quandl
quandl.ApiConfig.api_key = "uyqJwyzGBAwPxJ5Jfg4F"
df = quandl.get("WIKI/AMZN")
print("\nFirst five rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

df = df[['Adj. Close']]
print("\nFirst five rows of Adj. Close:")
print(df.head())

forecast_out = 30 # n days predicting window
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)
print("\nLast five rows of Adj. Close:")
print(df.tail())

# IV set
X = np.array(df.drop(['Prediction'], 1))
X = X[:-forecast_out]
X = ((X - X.mean())/(X.max() - X.min()))
# y = boston['target']
print("\nIndependent Variable X:")
print(X)


# DV set
y = np.array(df['Prediction'])
y = y[:-forecast_out]
print("\nDependent Variable X:")
print(y)

# common code
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33, random_state = 5)

X_train = X_train.T
y_train = np.array([y_train])
X_val = X_val.T
y_val = np.array([y_val])

linear_regression_model(X_train, y_train, X_val, y_val, 0.4, 500)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model = lr.fit(X_train.T, y_train.T)
predictions = lr.predict(X_val.T)
MAE_val_with_sklearn = (1 / y_val.shape[1]) * np.sum(np.abs(predictions - y_val.T))

print("MAE_val_with_sklearn:", MAE_val_with_sklearn)