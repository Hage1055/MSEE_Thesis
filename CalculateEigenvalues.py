# -*- coding: utf-8 -*-
"""
Author: Phillip Hagen

Description: Calculate the difference in Eigenvalues from a state to the 
             ground state
"""

import numpy as np
import matplotlib.pyplot as plt

hbar = 1.054e-34
h_nobar = 6.625e-34
melec = 9.11e-31
ecoul = -1.6e-19
eV2J = 1.6e-19
J2eV = 1/eV2J

plt.rcParams.update({'font.size': 20})

#%% Numerical Calculations

""" --- Numerical calculation of eigenenergies from state 1 - 5 --- """

a = 2*np.pi*35*1e-10 #Circumference of a torus with radius 35 A
n = 1 #eigenstate number
EE = np.zeros(5) #Eigen energy of each state
print('\nAnalytical calculations of eigenenergies\n')
for n in range(1,6):
    p = h_nobar/(a/n)
    E = np.round(J2eV*0.5*(p/melec)*p, 6)
    EE[n - 1] = E
    print('state', n, ':      E =', np.round(E, 6), 'eV')
    
#%% Simulation Results

""" --- Subtract simulated ground state eigen energy from simulated eigen
        eigen energies of each state 1 through 5 --- """
     
GroundStateEigenEnergy = 0.46456771456771456  #Simulated ground state energy
SETop = np.array([0.003308, 0.012198, 0.028532, 0.05024, 0.078152]) #Simulated energy values
SEE = np.zeros(5)
print('\nSimulated eigenenergies\n')
for n in range(1, 6):
    SEE[n - 1] = SETop[n - 1]
    print('state', n, ':      E_sim =', np.round(SEE[n - 1], 6), 'eV')

#%% Percent Difference

PerDiff = np.zeros(5) 
print('\nPercent difference\n')
for n in range(1, 6):
    PerDiff[n - 1] = (SEE[n - 1] - EE[n - 1])/np.abs(EE[n - 1])*100
    print('state', n, ':      Difference =', np.round(PerDiff[n - 1], 3), '%')
    
FracDiff = np.zeros(5)
for n in range(1, 6):
    FracDiff[n - 1] = (SEE[n - 1] - EE[n - 1])/np.abs(EE[n - 1])
    
#%% Plot

state = np.arange(1, 6, 1)
#plt.figure(figsize = (10, 7))
#plt.title('Numerical vs Simulated Eigen Energies')
#plt.ylabel('Eigen Energy [eV]')
#plt.xlabel('State')
#plt.plot(state, EE, label = 'Numerical', marker = '+', ms = '16')
#plt.plot(state, SEE, label = 'Analytical', marker = '.', ms = '12')
#plt.plot(state, np.abs(FracDiff), label = 'Fractional Difference', marker = 'x', 
#         ms = '16')
#plt.xticks(np.linspace(1, 5, 5))
#plt.grid()
#plt.legend()
#plt.show()

fig, ax1 = plt.subplots(figsize = (10, 7))
ax1.set_xlabel('State')
ax1.set_ylabel('Eigen Energy [eV]')
lns1 = ax1.plot(state, EE, label = 'Analytical', marker = '+', ms = '16')
lns2 = ax1.plot(state, SEE, label = 'Numerical', marker = '.', ms = '12')
ax1.set_xticks(np.linspace(1, 5, 5))
ax2 = ax1.twinx()
ax2.set_ylim([0, 1])
lns3 = ax2.plot(state, np.abs(FracDiff), label = 'Fractional Difference (Right)', 
         color = 'g', marker = 'x', ms = '16')
ax2.set_ylabel('Fractional Difference')
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
ax1.grid()
fig.tight_layout()
plt.show()