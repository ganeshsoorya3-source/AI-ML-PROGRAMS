import numpy as np
from sklearn.naive_bayes import GaussianNB
X=np.array([[1,20],[2,21],[3,22],[9,31],[10,32],[11,33]])
y=np.array([0,0,0,1,1,1])
model=GaussianNB()
model.fit(X, y)
prediction=model.predict([[12,34]])
print("Prdiction:",prediction)
