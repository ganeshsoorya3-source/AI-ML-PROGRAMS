import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

X = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 7],
    [7, 8],
    [8, 9]
])

y = np.array([0, 0, 0, 1, 1, 1])

model = SVC(kernel='linear')

model.fit(X, y)

prediction = model.predict([[4, 5]])

print("Prediction:", prediction)

plt.scatter(X[:, 0], X[:, 1], c=y)

plt.scatter(model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            facecolors='none',
            edgecolors='black',
            s=100)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Support Vector Machine")

plt.show()