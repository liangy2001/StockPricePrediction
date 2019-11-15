import numpy as np
from old_files.old_linear_models import LinearRegressionNormal

X = np.array([[0], [1], [2], [3]])
y = np.array([0, 1, 2, 3])
regr = LinearRegressionNormal()
regr.fit(X, y)
print(regr.w)
print(regr.score(X, y))