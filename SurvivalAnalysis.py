import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize

class SurvivalAnalysis:

    def sim(self, model, theta0, size, y_max):
        D = []
        for i in range(size):
            x1 = random.randint(0, 1)
            x2 = random.uniform(0, 1)
            myu = model([x1, x2], theta0)
            history = list(np.random.exponential(1 / myu, y_max))
            D += [[x1, x2] + history]
        D = pd.DataFrame(D)
        D.columns = ["x1", "x2"] + ["y" + str(i + 1) for i in range(y_max)]
        return D
    
    def LogLikelihood_(self, myu, history):
        LL = 0.0
        months_ = 0
        for h in range(len(history)):
            if (self.months - months_) >= history[h]: 
                LL += np.log(myu) - myu * (history[h]) 
            else: # corresponds to t_n ~ t_{n + 1}
                LL += - myu * (self.months - months_)
            months_ += history[h]
            if months_ > self.months:
                break
        return LL
    
    def NegativeMarginalLogLikelihood(self, theta):
        NMLL = 0
        for i in range(len(self.Y)):
            myu = self.model(self.X[i], theta)
            NMLL -= self.LogLikelihood_(myu, self.Y[i])
        return NMLL
    
    def cons(self, theta, ipshi=1e-5):
        return theta - ipshi
    
    def fit(self, model, X, Y, months, theta_init):
        self.model = model
        self.X = X
        self.Y = Y
        self.months = months
        cons = ({'type': 'ineq', 'fun': self.cons})
        theta_ = minimize(self.NegativeMarginalLogLikelihood, x0=theta_init, method='SLSQP', constraints=cons)
        return theta_["x"]