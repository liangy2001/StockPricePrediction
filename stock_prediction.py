import pandas as pd
import quandl
import math
import numpy as np
from linear_models import LinearRegressionCustomized
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

def calculate_sklearn_MAE(model, X_val, y_val):
    """calculate mean absolute error"""
    predictions = model.predict(X_val)
    return (1/y_val.shape[0]) * np.sum(np.abs(predictions - y_val))


# load data
style.use('ggplot')

quandl.ApiConfig.api_key = "uyqJwyzGBAwPxJ5Jfg4F"
df = quandl.get("WIKI/GOOGL")
print(df.head())

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())

forecast_col = 'Adj. Close' # select Adj. CLose as the column to forecast on
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out) # shift out the dates for prediction

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])


# ML: Customized Linear Regression
clf_LRC = LinearRegressionCustomized(eta=0.001, epochs=500, test_size=0.2, random_state=5)
clf_LRC_name = 'LinearRegressionCustomized'
clf_LRC.fit(X, y)
print("\nModel:", clf_LRC_name)
print("confidence:", clf_LRC.get_R2_score())
print("MAE: ", clf_LRC.MAE_val)



# data preprocessing for sklearn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# sklearn LR
clf_LR = LinearRegression(n_jobs=-1)
clf_LR_name = 'LinearRegression'
clf_LR.fit(X_train, y_train)
print("\nModel:", clf_LR_name)
print("confidence:", clf_LR.score(X_test, y_test))
print("MAE: ", calculate_sklearn_MAE(clf_LR, X_test, y_test))

# sklearn SVR
clf_SVR = SVR(kernel='linear')
clf_SVR_name = 'SVR'
clf_SVR.fit(X_train, y_train)
print("\nModel:", clf_SVR_name)
print("confidence:", clf_SVR.score(X_test, y_test))
print("MAE: ", calculate_sklearn_MAE(clf_SVR, X_test, y_test))

# forecast
#---------------------------------------------------------------
forecast_set = clf_LRC.predict(X_lately).flatten() # predict future price

# plot graph: price over time
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
plt.savefig((clf_LRC.filepath + 'forecast_plot.png').replace(' ', ''))

