# -*- coding: utf-8 -*-
"""
Author: Phillip Hagen

Description: Franctional difference of magnetic dipole moments
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})

#%% Data

B_amp = np.array([0, 10, 100, 500, 1000])

#%%% r_tube = 6

""" --- [0] is 0 T, [1] is 10 T, [2] is 100 T, [3] is 500 T, [4] is 1000 T --- """

State1Values6 = np.array([-0.9195436044, -0.9195491499, -0.9195706461, 
                          -0.9187756106, -0.914678861])
State2Values6 = np.array([-1.7895719056, -1.78954489, -1.7894630585, 
                          -1.7924490687, -1.8015563172])
State3Values6 = np.array([-2.7571348085, -2.7570954491, -2.7567298782, 
                          -2.7551463031, -2.7539519883])
State4Values6 = np.array([-3.6735889186, -3.6735816329, -3.6735359499, 
                          -3.6737375213, -3.6743537164])
State5Values6 = np.array([-4.5895883295, -4.5895941912, -4.5896529897, 
                          -4.5899690813, -4.5900310515])

#%%% r_tube = 4

State1Values4 = np.array([-0.908844191, -0.9088864926, -0.9091143438, 
                          -0.9048104031, -0.8797691499])
State2Values4 = np.array([-1.423705571, -1.42346326, -1.4227218422, 
                          -1.4492794922, -1.5304602608])
State3Values4 = np.array([-2.7302260743, -2.729896683, -2.7269048715, 
                          -2.7152923418, -2.7090016205])
State4Values4 = np.array([-3.6217143479, -3.6216334892, -3.6213450438, 
                          -3.6282948269, -3.6418695984])
State5Values4 = np.array([-4.5578958958, -4.5579230068, -4.5582669194, 
                          -4.56120944, -4.5632487032])

#%%% r_tube = 2

State1Values2 = np.array([-0.8631178741, -0.8630281741, -0.8622232541, 
                          -0.8586999221, -0.8544299941])
State2Values2 = np.array([-1.7312363802, -1.7311734789, -1.7306114914, 
                          -1.7282050859, -1.7254131692])
State3Values2 = np.array([-2.6157613311, -2.6157349845, -2.6155027089, 
                          -2.6145759693, -2.613659923])
State4Values2 = np.array([-3.5135326327, -3.513543845, -3.5136489703, 
                          -3.5142070112, -3.5151082653])
State5Values2 = np.array([-4.4232949509, -4.4233354621, -4.4237030477, 
                          -4.4253993749, -4.4276519255])

#%%% r_tube = 1

State1Values1 = np.array([-0.8824887991, -0.8824655136, -0.8822556483, 
                          -0.8813153531, -0.8801168537])
State2Values1 = np.array([-1.7605057168, -1.7604777666, -1.7602257714, 
                          -1.7590960274, -1.757659971])
State3Values1 = np.array([-2.637009552, -2.6369828101, -2.6367397042, 
                          -2.6355979434, -2.6339942023])
State4Values1 = np.array([-3.5071755017, -3.5071110123, -3.5065232876, 
                          -3.5037580752, -3.4999838327])
State5Values1 = np.array([-4.3527159078, -4.352625472, -4.3518167784, 
                          -4.34835995, -4.3444582326])

#%% Calculate fractional difference

State1Compare6 = np.zeros(len(State1Values6))
for i in range(1, 5):
    State1Compare6[i] = (State1Values6[i] - State1Values6[0])/np.abs(State1Values6[0])*100
    print('Fractional Change: ', State1Compare6[i])  

#%% Plotting

#%%% r_tube = 6

#%%%% State = 1

plt.figure(figsize=(9,6))
plt.plot(B_amp, State1Values6, 'o-')
#plt.title('$1^{st}$ excited state, $r_{tube}$ = 6')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$A\cdot\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState1tube6')
plt.show()

#%%%% State = 2

plt.figure(figsize=(9,6))
plt.plot(B_amp, State2Values6, 'o-')
#plt.title('$2^{nd}$ excited state, $r_{tube}$ = 6')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$A\cdot\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState2tube6')
plt.show()

#%%%% State = 3

plt.figure(figsize=(9,6))
plt.plot(B_amp, State3Values6, 'o-')
#plt.title('$3^{rd}$ excited state, $r_{tube}$ = 6')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$A\cdot\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState3tube6')
plt.show()

#%%%% State = 4

plt.figure(figsize=(9,6))
plt.plot(B_amp, State4Values6, 'o-')
#plt.title('$4^{th}$ excited state, $r_{tube}$ = 6')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$A\cdot\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState4tube6')
plt.show()

#%%%% State = 5

plt.figure(figsize=(9,6))
plt.plot(B_amp, State5Values6, 'o-')
#plt.title('$5^{th}$ excited state, $r_{tube}$ = 6')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$A\cdot\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState5tube6')
plt.show()

#%%% r_tube = 4

#%%%% State = 1

plt.figure(figsize=(9,6))
plt.plot(B_amp, State1Values4, 'o-')
plt.title('$1^{st}$ excited state, $r_{tube}$ = 4')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState1tube4')
plt.show()

#%%%% State = 2

plt.figure(figsize=(9,6))
plt.plot(B_amp, State2Values4, 'o-')
plt.title('$2^{nd}$ excited state, $r_{tube}$ = 4')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState2tube4')
plt.show()

#%%%% State = 3

plt.figure(figsize=(9,6))
plt.plot(B_amp, State3Values4, 'o-')
plt.title('$3^{rd}$ excited state, $r_{tube}$ = 4')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState3tube4')
plt.show()

#%%%% State = 4

plt.figure(figsize=(9,6))
plt.plot(B_amp, State4Values4, 'o-')
plt.title('$4^{th}$ excited state, $r_{tube}$ = 4')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState4tube4')
plt.show()

#%%%% State = 5

plt.figure(figsize=(9,6))
plt.plot(B_amp, State5Values4, 'o-')
plt.title('$5^{th}$ excited state, $r_{tube}$ = 4')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState5tube4')
plt.show()

#%%% r_tube = 2

#%%%% State = 1

plt.figure(figsize=(9,6))
plt.plot(B_amp, State1Values2, 'o-')
plt.title('$1^{st}$ excited state, $r_{tube}$ = 2')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState1tube2')
plt.show()

#%%%% State = 2

plt.figure(figsize=(9,6))
plt.plot(B_amp, State2Values2, 'o-')
plt.title('$2^{nd}$ excited state, $r_{tube}$ = 2')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState2tube2')
plt.show()

#%%%% State = 3

plt.figure(figsize=(9,6))
plt.plot(B_amp, State3Values2, 'o-')
plt.title('$3^{rd}$ excited state, $r_{tube}$ = 2')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState3tube2')
plt.show()

#%%%% State = 4

plt.figure(figsize=(9,6))
plt.plot(B_amp, State4Values2, 'o-')
plt.title('$4^{th}$ excited state, $r_{tube}$ = 2')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState4tube2')
plt.show()

#%%%% State = 5

plt.figure(figsize=(9,6))
plt.plot(B_amp, State5Values2, 'o-')
plt.title('$5^{th}$ excited state, $r_{tube}$ = 2')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState5tube2')
plt.show()

#%%% r_tube = 1

#%%%% State = 1

plt.figure(figsize=(9,6))
plt.plot(B_amp, State1Values1, 'o-')
plt.title('$1^{st}$ excited state, $r_{tube}$ = 1')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState1tube1')
plt.show()

#%%%% State = 2

plt.figure(figsize=(9,6))
plt.plot(B_amp, State2Values1, 'o-')
plt.title('$2^{nd}$ excited state, $r_{tube}$ = 1')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState2tube1')
plt.show()

#%%%% State = 3

plt.figure(figsize=(9,6))
plt.plot(B_amp, State3Values1, 'o-')
plt.title('$3^{rd}$ excited state, $r_{tube}$ = 1')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState3tube1')
plt.show()

#%%%% State = 4

plt.figure(figsize=(9,6))
plt.plot(B_amp, State4Values1, 'o-')
plt.title('$4^{th}$ excited state, $r_{tube}$ = 1')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState4tube1')
plt.show()

#%%%% State = 5

plt.figure(figsize=(9,6))
plt.plot(B_amp, State5Values1, 'o-')
plt.title('$5^{th}$ excited state, $r_{tube}$ = 1')
plt.grid(which = 'both')
plt.xlabel('$B_{amp}$ [T]')
plt.ylabel('Magnetic dipole moment [$\u212B^2$]')
plt.xscale('log')
plt.savefig('MdmState5tube1')
plt.show()