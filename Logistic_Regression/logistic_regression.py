import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X=np.array([[1],[2],[3],[4],[5],[6]])
y=np.array([0,0,0,1,1,1])
model=LogisticRegression()
model.fit(X,y)
prediction=model.predict([[3.5]])
print("Prediction:", prediction)
plt.scatter(X,y)
plt.plot(X, model.predict_proba(X)[:,1], color='red')
plt.xlabel("X")
plt.ylabel("Probability")
plt.title("Logistic Regression")
plt.show()
