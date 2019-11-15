import quandl
import numpy as np
#from sklearn.linear_model import LinearRegression
from old_files.old_linear_models import LinearRegression
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')
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
print("\nIndependent Variable X:")
print(X)


# DV set
y = np.array(df['Prediction'])
y = y[:-forecast_out]
print("\nDependent Variable X:")
print(y)

# split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

# training
lr = LinearRegression()
lr.fit(x_train, y_train)

# testing
# get r_2 score for testing set
lr_confidence = lr.score(x_test, y_test)
print("\nlr confidence (r^2): ", lr_confidence)
x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
print(x_forecast)

lr_prediction = lr.predict(x_forecast)
print(lr_prediction)