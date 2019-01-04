# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:05:06 2017

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Maziar Raissi, Hessam Babaee
"""

import sys
sys.path.insert(0, 'PGP/')


import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from parametric_GP import PGP
from Utilities import Normalize
from Utilities import Denormalize
import scipy.io
if __name__ == "__main__":
    
  

    data = scipy.io.loadmat('SSTData/TrainingData.mat')
    X = data["X_H"]
    y = data["Y_H"]

    data_star = scipy.io.loadmat('SSTData/XStarMap.mat')
    XStarMap = data_star["XStar"]

    data_star = scipy.io.loadmat('SSTData/XStarBuoy.mat')
    XStarBuoy = data_star["XStar"]

    Normalize_input_data = 1
    Normalize_output_data = 1
    

    # Normalize Input Data
    if Normalize_input_data == 1:
        X_m = np.mean(X, axis = 0)
        X_s = np.std(X, axis = 0)
        X = Normalize(X, X_m, X_s)
        XStarMap = Normalize(XStarMap, X_m, X_s)
        XStarBuoy = Normalize(XStarBuoy, X_m, X_s)

    # Normalize Output Data
    if Normalize_output_data == 1:
        y_m = np.mean(y, axis = 0)
        y_s = np.std(y, axis = 0)   
        y = Normalize(y, y_m, y_s)
        
       
    # Model creation
    M = 2000
    pgp = PGP(X, y, M, max_iter = 2000, N_batch = 1,
              monitor_likelihood = 100, lrate = 1e-3)
        
    # Training
    pgp.train()
    
    # Prediction
    mean_star, var_star = pgp.predict(XStarMap)
    mean_star = Denormalize(mean_star, y_m, y_s)
    std_star = Denormalize(var_star**.50, 0, y_s)
    scipy.io.savemat('SSTData/PredictionMap.mat', {'mean_star':mean_star,'std_star':std_star})

    mean_star, var_star = pgp.predict(XStarBuoy)
    mean_star = Denormalize(mean_star, y_m, y_s)
    std_star = Denormalize(var_star**.50, 0, y_s)
    scipy.io.savemat('SSTData/PredictionBuoy.mat', {'mean_star':mean_star,'std_star':std_star})

    
