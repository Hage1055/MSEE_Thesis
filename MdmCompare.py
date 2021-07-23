# -*- coding: utf-8 -*-
"""
@author: Phillip Hagen

Description: Plotting quantum vs. classical magnetic dipole moments
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Global variables

Bz = np.arange(-50.0, 75.0, 25.0)
state = np.arange(-10, 12, 2)

#%% Magnetic dipole moments

MdmC = [9.055,7.33,5.449,3.848,1.628,0,-1.628,-3.848,-5.449,-7.33,-9.055]
MdmQ = [9.036,7.134,5.454,3.589,1.619,0,-1.619,-3.589,-5.454,-7.134,-9.036]


#%% Plotting

plt.figure(figsize = (10, 7))
plt.subplot(211)
plt.title('Magnetic Dipole Moment Comparison')
plt.axis([-11.0, 11.0, -10.0, 10.0])
plt.ylabel('Magnetic Dipole Moment')
plt.xlabel('State')
plt.plot(state, MdmC, label = 'Classic', marker = '+', ms = '16')
plt.plot(state, MdmQ, label = 'Quantum', marker = '.', ms = '12')
plt.xticks(np.arange(-10, 12, 2))
plt.grid()
plt.legend()
plt.legend()
plt.show()