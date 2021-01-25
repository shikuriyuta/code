import numpy as np
import pandas as pd
import random

class SurvivalAnalysis:

    def sim(self, size, months):
        D = []
        for i in range(size):
            x1 = random.randint(0, 1)
            x2 = random.uniform(0, 1)
            lam = 6 * x1 + 6 * x2
            myu = 1/ lam
            history = [np.random.poisson(lam) for m in range(months)]
            D += [[x1, x2] + history]
        D = pd.DataFrame(D)
        D.columns = ["x1", "x2"] + ["y" + str(m + 1) for m in range(months)]
        return D
    
    def MarginalLogLikelihood(self, history, myu, months):
        MLL = 0.0
        for m in range(months):
            MLL += 
        #MLL = np.log(myu) + 
        #LL = sum([np.log(() for m in range(months)])
        return 1





