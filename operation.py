from SurvivalAnalysis import SurvivalAnalysis
import numpy as np

# parameters
months = 12 # observe period
def model(x, theta): # model (theta > 0)
    return theta * x[0] + theta * x[1]

# simulation dataset
theta0 = 0.3
size = 10000
y_max = months # maximum observations
D = SurvivalAnalysis().sim(model, theta0, size, y_max)
X = np.array(D[["x1", "x2"]])
Y = np.array(D[["y" + str(i + 1) for i in range(y_max)]])

# main
theta_init = [0.1] # initial value for optimization
theta = SurvivalAnalysis().fit(model, X, Y, months, theta_init)
print(theta)