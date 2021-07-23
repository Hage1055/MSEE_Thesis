# -*- coding: utf-8 -*-
"""
@author: Phillip Hagen

Description: #rd harmonic term to B_Max cubed
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Global variables

n_theta = np.arange(20, 8, -2)
B_max_30 = 7.5
B_max_20 = 5.0

#%% Third harmonic terms

Harmonic3_30 = np.array([2e-8, 2.15e-6, 4.01e-6, 7.49e-6, 1.36e-5, 2.086e-5])
B_max_cubed_30 = B_max_30**3
    
Parameter_30 = Harmonic3_30/B_max_cubed_30

Harmonic3_20 = np.array([3.4e-7, 6.9e-7, 1.19e-6, 2.26e-6, 4.03e-6, 6.17e-6])
B_max_cubed_20 = B_max_20**3

Parameter_20 = Harmonic3_20/B_max_cubed_20

#%% Print and plot values

print('Values when B_amp = 30', Parameter_30)
print('Values when B_amp = 20', Parameter_20)
plt.figure(figsize = (10, 7))
plt.title('Third harmonic divided by B_max cubed')
plt.xlabel('n_grate')
plt.ylabel('harmonic/B^3')
plt.plot(n_theta, Parameter_30, label = '30 T', marker = '.', ms = '16')
plt.plot(n_theta, Parameter_20,label = '20 T',  marker = '+', ms = '16')
plt.grid()
plt.legend()
plt.show()