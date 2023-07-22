import pandas as pd
from sklearn import linear_model

X, y = [2, 3, 4, 5, 6], [70, 75, 85, 90, 95]

X, y = pd.DataFrame(X), pd.DataFrame(y)

reg = linear_model.LinearRegression()

reg.fit(X, y)

print(reg.predict([[7]]))