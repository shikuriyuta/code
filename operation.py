from SurvivalAnalysis import SurvivalAnalysis
import numpy as np

theta0 = 0.7
size = 10000
y_max = 5
months = 12
SA = SurvivalAnalysis()
D = SA.sim(theta0, size, y_max)
X = np.array(D[["x1", "x2"]])
Y = np.array(D[["y" + str(i + 1) for i in range(y_max)]])
theta = SA.fit(X, Y, months)
print(theta)
