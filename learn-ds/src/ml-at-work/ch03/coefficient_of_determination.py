from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x, y)

from sklearn.metrics import r2_score
r2 = r2_score(y, lr.predict(x))