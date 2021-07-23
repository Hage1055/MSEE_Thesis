# -*- coding: utf-8 -*-
"""
Author: Phillip Hagen

Description: Plot Torus in 3d
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({'font.size': 16})

#%% Define torus

n = 50

theta = np.linspace(0, 2*np.pi, n)
phi = np.linspace(0, 2*np.pi, n)
theta, phi = np.meshgrid(theta, phi)
c, a = 35, 6
x = (c + a*np.cos(theta)) * np.cos(phi) 
y = (c + a*np.cos(theta)) * np.sin(phi) 
z = a * np.sin(theta) 

#%% Plotting
#%%% Without magnetic field

fig = plt.figure(figsize = (9, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.tick_params(axis = 'x', pad = 5)
ax1.tick_params(axis = 'y', pad = 5)
ax1.tick_params(axis = 'z', pad = 5)
ax1.set_xlim(-50, 50)
ax1.set_ylim(-50, 50)
ax1.set_zlim(-50, 50)
ax1.set_xlabel('\n\nY [\u212B]')
ax1.set_ylabel('\n\nX [\u212B]')
ax1.zaxis.set_rotate_label(False)
ax1.set_zlabel('Z [\u212B]\n\n', rotation = 90)
ax1.plot_wireframe(x,y,z,rstride=1,cstride=5,color='w',edgecolor='c',
                 linewidth=.4, zorder = -1)
ax1.view_init(25, 45)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.tick_params(axis = 'x', pad = 5)
ax2.tick_params(axis = 'y', pad = 5)
ax2.tick_params(axis = 'z', pad = 18)
ax2.set_xlim(-50, 50)
ax2.set_ylim(-50, 50)
ax2.set_zlim(-50, 50)
ax2.set_xlabel('\n\nCells [\u212B]')
ax2.set_zlabel('Cells [\u212B]\n\n\n')
ax2.plot_wireframe(x, y, z, rstride=1, cstride=5, color='w', edgecolor='c',
                 linewidth=.4, zorder = -1)
ax2.view_init(0, 90)
ax2.set_yticks([])
plt.savefig('Torus3D.png')
plt.show()

#%%% With applied magnetic field

fig = plt.figure(figsize = (9, 6))
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
ax1.tick_params(axis = 'x', pad = 5)
ax1.tick_params(axis = 'y', pad = 5)
ax1.tick_params(axis = 'z', pad = 5)
ax1.set_xlim(-50, 50)
ax1.set_ylim(-50, 50)
ax1.set_zlim(-50, 50)
ax1.set_xlabel('\n\nY [\u212B]')
ax1.set_ylabel('\n\nX [\u212B]')
ax1.zaxis.set_rotate_label(False)
ax1.set_zlabel('Z [\u212B]\n\n', rotation = 90)
ax1.plot_wireframe(x,y,z,rstride=2,cstride=2,color='c',
                 linewidth=.3, zorder = -1)
ax1.view_init(25, 45)
ax1.quiver(0, 0, 0, 0, 0, 60, color = 'm', linewidth = 5, zorder = 2)
ax1.text(-10, 5, 60, '\n$B_0$', color = 'm', fontsize = '20')
plt.savefig('TorusWithBField3D.png')
plt.show()