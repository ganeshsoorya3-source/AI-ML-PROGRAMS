import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X=np.array([[1],[2],[3],[4],[5]])
y=np.array([2,4,6,8,10])
model=LinearRegression()
model.fit(X,y)

prediction=model.predict([[6]])

print("Prediction:",prediction)
print("Slope:",model.coef_)
print("Intercept:",model.intercept_)

plt.scatter(X,y)
plt.plot(X,model.predict(X))
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.show()
