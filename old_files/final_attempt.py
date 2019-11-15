from linear_models import LinearRegressionCustomized
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

clf = LinearRegressionCustomized(test_size=0.2)
print(clf.fit(X, y).get_R2_score())

