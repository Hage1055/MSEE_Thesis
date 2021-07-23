"""Se3d_nytorus.py
from Se3d_LGTOR6a.py.
"""

#%% Package imports

from math import pi, sqrt, atan2
from cmath import exp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from numba import jit
from math import exp, cos, sin
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import time

#%% Global variables

NN = 100
MM = 100
KK = 30

NC = int(NN / 2)
MC = int(MM / 2)
KC = int(KK / 2)

dims = (NN, MM, KK)
hbar = 1.054e-34
h_nobar = 4.135e-15  # in eV
meff = 9.1e-31
melec = 9.1e-31
ecoul = -1.6e-19
eV2J = 1.6e-19
J2eV = 1 / eV2J

del_x = 1e-10
#dt = 0.08e-15
dt = 0.02e-16
DT = dt * 1e15
ra = (0.5 * hbar / melec) * (dt / del_x ** 2)
rd = dt / hbar
DX = del_x * 1e9

XX = np.linspace(DX, DX * NN, NN)
YY = np.linspace(DX, DX * MM, MM)
X, Y = np.meshgrid(XX, YY)

xpml = np.ones(NN)
ypml = np.ones(MM)
zpml = np.ones(KK)

""" The max number of iterations"""
T_tot = 100000

Btime = np.zeros(T_tot)
MDM_time = np.zeros(T_tot)

TT = np.linspace(0, DT * T_tot, T_tot)
Ptime_rl = np.zeros(T_tot)
Ptime_im = np.zeros(T_tot)

xpos = np.zeros(T_tot)
ypos = np.zeros(T_tot)

""" F_tot is the size of the FFT buffer"""

F_tot = 1000000

del_F = 1e-12 / (F_tot * dt)  # in THz
FF = np.linspace(0, del_F * (F_tot), F_tot)
print('del_F = ',del_F,' THz')

del_E = 1e12 * h_nobar * del_F
EE = np.linspace(0, del_E * F_tot, F_tot)
print('del_E = ',del_E,' eV')

prl = np.zeros((dims))
pim = np.zeros((dims))

Phi_rl = np.zeros((dims))
Phi_im = np.zeros((dims))

""" This specifies the number of steps used to create
    an eigenfunction once the energy is known."""
PF_TIME = 10000
omegaT = 0
win_Phi = np.zeros(PF_TIME)
for n in range(PF_TIME):
    win_Phi[n] = .5 * (1 - cos(2 * pi * n / PF_TIME))

plt.subplot(311)
plt.plot(win_Phi)
plt.title('Window for PHI')
plt.grid()
plt.tight_layout()
plt.show()

B0 = 0

#%% B-field input

Bamp = float(input('Bamp (Tesla)  --> '))
print('Bamp =', round(Bamp, 3))

if Bamp >= 0.01:
    Bfreq = float(input('B freq (THz) --> '))
    B_TIME = T_tot

    for n in range(B_TIME):
        Btime[n] = (Bamp * .5 * (1 - cos(2 * pi * n / B_TIME))
                    * cos(2 * pi * Bfreq * 1e12 * dt * (n - .5 * B_TIME)))

    plt.subplot(311)
    plt.plot(TT, Btime)
    plt.axis([0, B_TIME * DT, -Bamp, Bamp])
    plt.xlabel('T (fs)')
    plt.grid()
    plt.tight_layout()
    plt.show()

Kmdm = ecoul * hbar / (2 * meff)  # Mag dipole moment constant
KdelB = (ecoul ** 2) / (4 * meff)  # B field magnetic dipole
K_dfB = dt * ecoul / (2 * meff)  # For the HO_B term; doesn't need rd

#%% Function definitions

@jit
def mag_dipole(prl, pim):
    mdm = 0 + 1j * 0

    for n in range(1, NN - 1):
        for m in range(1, MM - 1):
            for k in range(KK):
                pmag = prl[n, m, k] ** 2 + pim[n, m, k] ** 2
                der_y = 0.5 * (n - NC) * (prl[n, m + 1, k] - prl[n, m - 1, k]
                                          + 1j * (pim[n, m + 1, k] - pim[n, m - 1, k]))
                der_x = 0.5 * (m - MC) * (prl[n + 1, m, k] - prl[n - 1, m, k]
                                          + 1j * (pim[n + 1, m, k] - pim[n - 1, m, k]))
                # The 0.5 are because the spatial derivative is over two cells.
                mdm = mdm - (Kmdm * (1j * prl[n, m, k] + pim[n, m, k]) * (der_y - der_x)
                             - B0 * KdelB * (del_x ** 2) * ((n - NC) ** 2 + (m - MC) ** 2) * pmag)

    return mdm


""" The 2D HO b field potential """


def HO_B(ecoul, meff, del_x):
    VoB = np.zeros((NN, MM))
    K_hoB = (ecoul ** 2) * (del_x ** 2) / (8 * meff)

    for n in range(NN):
        for m in range(MM):
            VoB[n, m] = K_hoB * ((n - NC) ** 2 + (m - MC) ** 2)

    return VoB

def cal_ptot(prl, pim):
    ptot = (np.sum(prl ** 2 + pim ** 2))
    print('ptot = ', round(ptot, 4))
    return ptot

def normal(prl, pim):
    pnorm = sqrt(np.sum(prl ** 2 + pim ** 2))
    prl = prl / pnorm
    pim = pim / pnorm
    ptot = np.sum(prl ** 2 + pim ** 2)
    return prl, pim, ptot

def surf_2D_rl(X, Y, PPP, NNN):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(Y, X, PPP[:, :, KC], rstride=10,
                           cstride=1, cmap=cm.coolwarm, linewidth=.4)
    ax.view_init(elev=60., azim=45)
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.title(NNN)
    plt.savefig('prl.png')
    plt.tight_layout()
    plt.show()

def surf_2D_im(X, Y, PPP, NNN):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(Y, X, PPP[:, :, KC], rstride=10,
                           cstride=1, cmap=cm.PuOr, linewidth=.4)
    ax.view_init(elev=45., azim=45)
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.title(NNN)
    plt.savefig('pim.png')
    plt.tight_layout()
    plt.show()

# @jit
def observ(prl, pim, V, del_x):

    PE = 0.
    for n in range(NN):
        for m in range(MM):
            for k in range(KK):
                PE = PE + V[n, m, k] * (prl[n, m, k] ** 2 + pim[n, m, k] ** 2)

    psi = np.zeros(dims, dtype=complex)
    for n in range(NN):
        for m in range(MM):
            for k in range(KK):
                psi[n, m, k] = prl[n, m, k] + 1j * prl[n, m, k]

    ke = 0. + 1j * 0

    for n in range(1, NN - 1):
        for m in range(1, MM - 1):
            for k in range(1, KK - 1):
                lap_psi = (-6 * psi[n, m, k]
                           + psi[n + 1, m, k] + psi[n - 1, m, k]
                           + psi[n, m + 1, k] + psi[n, m - 1, k]
                           + psi[n, m, k - 1] + psi[n, m, k + 1])
                lap_prl = (-6 * prl[n, m, k]
                           + prl[n + 1, m, k] + prl[n - 1, m, k]
                           + prl[n, m + 1, k] + prl[n, m - 1, k]
                           + prl[n, m, k - 1] + prl[n, m, k + 1])
                lap_pim = (-6 * pim[n, m, k]
                           + pim[n + 1, m, k] + pim[n - 1, m, k]
                           + pim[n, m + 1, k] + pim[n, m - 1, k]
                           + pim[n, m, k - 1] + pim[n, m, k + 1])
                ke = ke + (lap_prl * prl[n, m, k] + lap_pim * pim[n, m, k]
                           + 1j * (-lap_prl * pim[n, m, k] + lap_pim * prl[n, m, k]))

    print('ke = ', ke)

    KE = -(hbar / del_x) ** 2 / (2 * meff) * ke.real

    print("PE = ", round(J2eV * PE, 4), " eV",
          "KE = ", round(J2eV * KE.real, 4), " eV")
    E = J2eV * (PE + KE.real)
    print("E  = ", round(J2eV * (PE + KE.real), 8), " eV")
    
    Tperiod = (4.135e-15/E)*(1/dt)
    print('Tperiod = ',round(Tperiod,1))

    return PE, KE

@jit
def fdtd(prl, pim, nsteps, T, xpml, ypml, zpml, Ptime_rl, Ptime_im,
         Phi_rl, Phi_im, omegaT, win_Phi, PF_TIME, xpos, ypos, MDM_time):
    print('nsteps = ', nsteps)
    print('B0 = ', B0)
    for _ in range(nsteps):
        T = T + 1

        for n in range(1, NN - 1):
            for m in range(1, MM - 1):
                for k in range(1, KK - 1):
                    prl[n, m, k] = (xpml[n] * ypml[m] * zpml[k] * prl[n, m, k]
                                    + rd * V[n, m, k] * pim[n, m, k]
                                    - ra * (-6 * pim[n, m, k]
                                            + pim[n + 1, m, k] + pim[n - 1, m, k]
                                            + pim[n, m + 1, k] + pim[n, m - 1, k]
                                            + pim[n, m, k - 1] + pim[n, m, k + 1]
                                            )
                                    + rd * ((B0 + Btime[T]) ** 2) * VoB[n, m] * pim[n, m, k]
                                    + K_dfB * (B0 + Btime[T]) * 0.5 * (-(m - MC) * (prl[n + 1, m, k] - prl[n - 1, m, k])
                                                                       + (n - NC) * (prl[n, m + 1, k] - prl[n, m - 1, k]))
                                    )

        for n in range(1, NN - 1):
            for m in range(1, MM - 1):
                for k in range(1, KK - 1):
                    pim[n, m, k] = (xpml[n] * ypml[m] * zpml[k] * pim[n, m, k]
                                    - rd * V[n, m, k] * prl[n, m, k]
                                    + ra * (-6 * prl[n, m, k]
                                            + prl[n + 1, m, k] + prl[n - 1, m, k]
                                            + prl[n, m + 1, k] + prl[n, m - 1, k]
                                            + prl[n, m, k - 1] + prl[n, m, k + 1]
                                            )
                                    - rd * ((B0 + Btime[T]) ** 2) * VoB[n, m] * prl[n, m, k]
                                    + K_dfB * (B0 + Btime[T]) * 0.5 * (-(m - MC) * (pim[n + 1, m, k] - pim[n - 1, m, k])
                                                          + (n - NC) * (pim[n, m + 1, k] - pim[n, m - 1, k]))
                                    )

        Ptime_rl[T] = prl[NC + 35, MC, KC]
        Ptime_im[T] = pim[NC + 35, MC, KC]
 
        for n in range(NN):
            for m in range(MM):
                for k in range(KK):
                    xpos[T] = xpos[T] + (n - NC) * (prl[n, m, k] ** 2 + pim[n, m, k] ** 2)
                    ypos[T] = ypos[T] + (m - MC) * (prl[n, m, k] ** 2 + pim[n, m, k] ** 2)
                #        print(T,'xpos,ypos:',round(xpos[T],3),round(ypos[T],3))

        #        This section calculates the magnetic dipole moment

        mdm = 0 + 1j * 0

        for n in range(1, NN - 1):
            for m in range(1, MM - 1):
                for k in range(KK):
                    pmag = prl[n, m, k] ** 2 + pim[n, m, k] ** 2
                    der_y = 0.5 * (n - NC) * (prl[n, m + 1, k] - prl[n, m - 1, k]
                                              + 1j * (pim[n, m + 1, k] - pim[n, m - 1, k]))
                    der_x = 0.5 * (m - MC) * (prl[n + 1, m, k] - prl[n - 1, m, k]
                                              + 1j * (pim[n + 1, m, k] - pim[n - 1, m, k]))
                    # The 0.5 are because the spatial derivative is over two cells.
                    mdm = mdm + (Kmdm * (1j * prl[n, m, k] + pim[n, m, k]) * (der_y - der_x)
                                 + (B0 + Btime[T]) * KdelB * (del_x ** 2) * ((n - NC) ** 2 + (m - MC) ** 2) * pmag)

        MDM_time[T] = 1e23 * mdm.real

        #      This section constructs an eigenfunction

        if T < PF_TIME:

            cos_term = win_Phi[T] * cos(omegaT * T)
            sin_term = win_Phi[T] * sin(omegaT * T)
            for n in range(NN):
                for m in range(MM):
                    for k in range(KK):
                        Phi_rl[n, m, k] = (Phi_rl[n, m, k]
                                           + cos_term * prl[n, m, k] - sin_term * pim[n, m, k])
                        Phi_im[n, m, k] = (Phi_im[n, m, k]
                                           + sin_term * prl[n, m, k] + cos_term * pim[n, m, k])
    #
    return prl, pim, T, Ptime_rl, Ptime_im, Phi_rl, Phi_im, xpos, ypos, MDM_time


@jit
def torus3D(V, r_torus, r_tube, n_theta):
    LL = 9
    L2 = int(LL / 2)

    Vgrate = np.zeros( (NN,MM) )
    
    for n in range(NN):
        for nn in range(LL):
            xdist = (NC - n) + (1 / LL) * (L2 - nn)
            for m in range(MM):
                for mm in range(LL):
                    ydist = (MC - m) + (1 / LL) * (L2 - mm)
                    for k in range(KK):
                        for kk in range(LL):
                            zdist = (KC - k) + (1 / LL) * (L2 - kk)
                            dist = np.sqrt(xdist ** 2 + ydist ** 2)
                            Vsum = (r_torus - dist) ** 2 + zdist ** 2

                            if r_tube ** 2 > Vsum:
                                theta = np.arctan(ydist / (xdist + 1e-7))
                                amp = cos(n_theta * theta)
                                if n_theta != 0:
                                    Vgrate[n,m] =( .065 * amp) / (LL ** 3)
                                V[n,m,k] = V[n,m,k] - 1/(LL**3) - Vgrate[n,m]
      
    plt.subplot(221)
    plt.contour(Vgrate) 
    plt.grid()
    plt.text(15,65, '$ r_torus %$= {}'.format(round(r_torus, 2)))
    plt.text(15,45, '$ r_tube   %$= {}'.format(round(r_tube, 2)))
    plt.text(15,25, '$ n theta %$= {}'.format(round(n_theta, 0)))
    plt.savefig('cont.png')
    plt.tight_layout()
    plt.show()                       #

    return V, Vgrate

""" The 2D HO b field potential """


def HO_B(ecoul, meff, del_x):
    VoB = np.zeros((NN, MM))
    #    K_hoB = ((ecoul*(del_x**2))/meff)*(ecoul/8)
    K_hoB = (ecoul ** 2) * (del_x ** 2) / (8 * meff)

    for n in range(NN):
        for m in range(MM):
            VoB[n, m] = K_hoB * ((n - NC) ** 2 + (m - MC) ** 2)

    return VoB

@jit
def mk_PML(npml):
    xpml = np.ones(NN)
    ypml = np.ones(MM)
    zpml = np.ones(KK)
    for n in range(npml + 1):
        xxn = (npml - n) / npml
        xpml[n] = 1 - .25 * xxn ** 3
        xpml[NN - 1 - n] = xpml[n]
    for n in range(npml + 1):
        xxn = (npml - n) / npml
        ypml[n] = 1 - .25 * xxn ** 3
        ypml[MM - 1 - n] = ypml[n]
    for n in range(npml + 1):
        xxn = (npml - n) / npml
        zpml[n] = 1 - .25 * xxn ** 3
        zpml[KK - 1 - n] = zpml[n]

    return xpml, ypml, zpml

@jit
def init_ring(r_torus):
    ptot = 0
    Vpos = np.zeros(dims)
    prl = np.zeros(dims)
    pim = np.zeros(dims)
    for n in range(NN):
        for m in range(MM):
            for k in range(KK):
                Vpos[n, m, k] = (r_torus-np.sqrt((n-NC)**2 + (m-MC) **2))**2 + (k-KC) ** 2
                prl[n, m, k] = exp(-Vpos[n, m, k] / 20)

    return prl, pim

#%% Main program

""" Initialize the  potential """

r_torus = 35
r_tube = 2
V = np.ones(dims)
n_theta = int(input('n_theta --> '))
V, Vgrate = torus3D(V, r_torus, r_tube, n_theta)
VoB = HO_B(ecoul, meff, del_x)
V = 4.6 * eV2J * V

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y,  J2eV*Vgrate)
plt.title('Toroidal Potential (XY plane)')
#plt.savefig('V_Tor.png')
plt.tight_layout()
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, J2eV * V[:, :, KC])
plt.title('Toroidal Potential (XY plane)')
plt.savefig('V_Tor.png')
plt.tight_layout()
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, rd * VoB[:, :])
plt.title('HO Potential (XY plane)')
plt.savefig('VoB.png')
plt.tight_layout()
plt.show()
  
plt.subplot(131)
plt.pcolor(V[NC, :, :])
plt.subplot(133)
plt.contour(V[NC, :, :], 1)
plt.grid()
plt.axis([0,30,0,30])
plt.tight_layout()
plt.show()

""" -- Initialize the waveform -----------"""

st_n = 0

III = int(input('1: initial pulse; 2: eigenfunction; 3. get PHI --> '))

""" Initialize a pulse to find the eigenfrequency """
if III == 1:

    prl, pim = init_ring(r_torus)
    
    plt.subplot(321)
    plt.plot(prl[:,MC,KC],'o-m', label = 'Pulse')
    plt.plot(0.2*J2eV*V[:,MC,KC],'k--', label = 'Potential')
    plt.axis([8,20,0,1.2])
    plt.xticks([9,11,13,15,17])
    plt.grid()
    
    plt.subplot(322)
    plt.plot(prl[:,MC,KC],'o-m', label = 'Pulse')
    plt.plot(0.2*J2eV*V[:,MC,KC],'k--', label = 'Potential')
    plt.axis([NN-20,NN-7,0,1.2])
#    plt.xticks([9,11,])
    plt.legend()
    plt.grid()
    
    plt.savefig('pulse.png')
    plt.tight_layout()
    plt.show()

""" Determine the eigenfunction """
if III == 2:
    E_eigen = input('Eigenenergy (eV) --> ')
    freq_0 = float(E_eigen) / h_nobar
    omegaT = 2 * pi * freq_0 * dt

    prl, pim = init_ring(r_torus)

""" Initialize with the eigenstate """
if III == 3:

    ph0_rl = np.load('Phillip_Phi_rl.npy')
    ph0_im = np.load('Phillip_Phi_im.npy')

    st_n = int(input('Initialize in state ---> '))
    print('st_n = ', st_n)
#    st_n = 0
    
    for n in range(NN):
        xdist = n - NC
        for m in range(MM):
            ydist = m - MC        
            theta = atan2(ydist, xdist)
            xxx = 1
            for k in range(KK):
                prl[n, m, k] = np.cos(st_n * theta) * ph0_rl[n, m, k] * xxx
                pim[n, m, k] = np.sin(st_n * theta) * ph0_rl[n, m, k] * xxx

prl, pim, ptot = normal(prl, pim)

surf_2D_rl(X, Y, prl, 'Initial')

PE, KE = observ(prl, pim, V, del_x)

""" Add the PML """
npml = 5
xpml,ypml,zpml = mk_PML( npml )

""" The core FDTD loop """

T = 0

while True:

    nsteps = int(input('How many time steps --> '))
    if nsteps == 0:
        break

    tic = time.time()    

    prl, pim, T, Ptime_rl, Ptime_im, Phi_rl, Phi_im, xpos, ypos, MDM_time = fdtd(
        prl, pim, nsteps, T, xpml, ypml, zpml, Ptime_rl, Ptime_im,
        Phi_rl, Phi_im, omegaT, win_Phi, PF_TIME, xpos, ypos, MDM_time)
    
    toc = time.time()
    print('FDTD time = ',round(toc - tic, 3))
    
    surf_2D_rl(X, Y, prl, 'prl')
    surf_2D_im(X, Y, pim, 'pim')

    plt.contour(X, Y, prl[:, :, KC])
    plt.contour(X, Y, pim[:, :, KC])

# Find the eigenenergy
    
    tic = time.time()
    
    PP = Ptime_rl - 1j * Ptime_im
    han = np.zeros(T)
    for n in range(T):
        han[n] = 0.5 * (1 - cos(2 * pi * n / T))
        PP[n] = han[n] * PP[n]

    pmax = max(Ptime_rl)
    pmin = min(Ptime_rl)
    plt.subplot(311)
    plt.plot(PP.real, 'k--')
    plt.title('Se3d_nytorus')
    plt.axis([0, T, pmin, pmax])
    plt.ylabel('PP(t)')
    plt.tight_layout()
    plt.grid()

    PF = (1 / NN) * np.fft.fft(PP, F_tot)
    PFabs = abs(PF)

    PFmax = max(PFabs)
    Ptop = 0
    Etop = 0 
    Etot = 0
    if T > 1000:
        for m in range(10000):
            if (PFmax-PFabs[m] ) < .01*PFmax:
                mmax = m
#                print('PFabs[m] = ',round(PFabs[m],4))
#                print('m = ',m,'   EE[m] = ',EE[mmax])
                if PFabs[m] > Ptop:
                    Ptop = PFabs[m]
                    Etop = EE[mmax]
                             
    print('Etop = ',Etop)
    
    plt.subplot(312)
    plt.plot(EE, PFabs, 'k')
    plt.axis([.5*Etop, 1.5*Etop,0, 1.2 * PFmax])
    plt.xticks([ 0.5*Etop,Etop , 1.5*Etop])    
    plt.text(.6*Etop, .5*PFmax, '$ KE %$= {}'.format(round(J2eV*KE, 4)))
    plt.text(.6*Etop, .2*PFmax, '$ PE %$= {}'.format(round(J2eV*PE, 4)))
    plt.text(.6*Etop, .8*PFmax, '$ Etop %$= {}'.format(round(Etop, 8)))
    plt.text(.9*Etop, .2*PFmax, '$ Etot %$= {}'.format(round(J2eV*(KE+PE), 4)))    
    plt.text(1.2*Etop, .8*PFmax, '$ T %$= {}'.format(round(T, 0)))
    plt.text(1.2*Etop, .4*PFmax, '$ st-n %$= {}'.format(round(st_n, 0)))

    plt.xlabel('E (eV)')
    plt.ylabel('PF(E)')
    plt.grid()
   
    plt.subplot(325)
    plt.contour(V[:,:,KC]) 
    plt.grid()
    plt.text(40,65, '$ r torus %$= {}'.format(round(r_torus, 0)))
    plt.text(40,45, '$ r tube %$= {}'.format(round(r_tube, 0)))
    plt.text(40,25, '$ n theta %$= {}'.format(round(n_theta, 0)))
    plt.savefig('tor_con.png')
    
    plt.subplot(326)
    plt.contour(prl[:,:,KC],cmap='bone')
    plt.grid()
    plt.savefig('FT.png')
    plt.tight_layout()
    plt.show()     
    
    toc = time.time()
    print('FFT time = ',round(toc - tic, 3))

    PE, KE = observ(prl, pim, V, del_x)

#    xpos, ypos = find_pos(prl,pim)
#    print('xpos = ',round(xpos),"  ypos = ",round(ypos))

    ptot = cal_ptot(prl, pim)
    print("T = ", T)
    
    E1 = .4471
    E0 = .4445
    freq_e01  = 1e-12*(E1 - E0)/h_nobar
    print('freq_e01 = ',round(freq_e01,4), 'THz')
    
    

    if III == 2:
        np.save('Phillip_Phi_rl', Phi_rl)
        np.save('Phillip_Phi_im', Phi_im)
        np.save('Phillip_V', V)

    if III == 3:
        mdm = 1e3 * mag_dipole(prl, pim)
        print('mdm = ', mdm)
        print('mdm.real = ', mdm.real)
        #  The 1e20 is because the units are Angstom squared
        Mdm_Q = (mdm.real * 1e20)
        print('Mdm (FDTD)= ', round(Mdm_Q, 6))

        T_period = 96e-15
        I_curr = ecoul / T_period  # 'ecoul' is already negative.
        print("I_curr = ", I_curr)
        Mdm_C = 1e3 * 1e20 * (I_curr) * pi * (r_torus * del_x) ** 2
        print('Mdm_C = ', round(Mdm_C, 3), ' A**2')

        plt.subplot(311)
        plt.plot(TT, xpos, 'b', label='X pos')
        plt.plot(TT, ypos, 'm--', label='Y pos')
        plt.title('Se3d_nytorus')
        #    plt.legend()
        plt.xlabel('Time (fs)')
        plt.grid()
        plt.text(20, 10, '$ st n %$= {}'.format(round(st_n, 1)))
        plt.text(20, -10, '$ B0 %$= {}'.format(round(B0, 1)))
        plt.text(5, -35, '$ Mdm_Q %$= {}'.format(round(Mdm_Q, 3)))
        plt.text(70, -35, '$ Mdm_C %$= {}'.format(round(Mdm_C, 3)))
        plt.axis([0, T * DT, -40, 30])
        plt.xticks([0, 96, 133.])
        plt.savefig('Pos.png')
        plt.tight_layout()
        plt.show()

        mmax = max(MDM_time)
        mmin = min(MDM_time)

        if Bamp >= 0.01:
            plt.subplot(221)
            plt.plot(TT, Btime, 'm', label='B time')
            plt.title('Se3d_LGTOR6')
            plt.axis([0, B_TIME * dt * 1e15, -Bamp, 1.5 * Bamp])
            plt.legend()
            plt.grid()

            BF = 4*(1 / F_tot) * abs(np.fft.fft(Btime, F_tot))
            Bmax = max(BF)

            plt.subplot(223)
            plt.plot(FF, BF)
            plt.yscale("log")
            plt.text(.5, .001, '$ Bmax %$= {}'.format(round(Bmax, 2)))
            plt.grid()
            plt.axis([0, 80, 1e-10, 2*Bmax])
#            plt.xticks([0, .7, 2.1, 3])

            np.save("MDM_time_Phillip", MDM_time)
            print(f"dt: {dt}, ddx: {del_x}")
            plt.subplot(222)
            plt.plot(TT, MDM_time, 'k', label='MDM time')
            plt.legend()
            plt.grid()

            MF = 4*(1 / F_tot) * abs(np.fft.fft(MDM_time, F_tot))
            MDmax = max(MF)
            
            n_1st = int(Bfreq/del_F)
            n_3rd = int(3*Bfreq/del_F)
#            n_1st = 14
#            n_3rd = 42
            MD_1st = MF[n_1st]
            MD_3rd = MF[n_3rd]

            plt.subplot(224)
            plt.plot(FF, MF)
            plt.yscale("log")
            plt.text(.4, .03, '$ 1st %$= {}'.format(round(MD_1st, 10)))
            plt.text(.4, .0005, '$ 3rd %$= {}'.format(round(MD_3rd, 15)))
            plt.grid()
            plt.axis([0, 80, 1e-8, 10])
            plt.xticks([10,20,60])
            plt.savefig('MDM.png')
            plt.tight_layout()
            plt.show()
            
            plt.subplot(111)
            plt.contour(V[:,:,KC]) 
            plt.grid()
            plt.text(40,50, '$ r torus %$= {}'.format(round(r_torus, 0)))
            plt.text(40,40, '$ r tube %$= {}'.format(round(r_tube, 0)))
            plt.text(40,30, '$ n theta %$= {}'.format(round(n_theta, 0)))
            plt.savefig('tor_con.png')
            
            plt.subplot(221)
            plt.contour(V[NC, :, :],1)
            plt.grid()
            plt.ylabel('z')
            plt.xlabel('y')
            plt.axis([0,30,0,30])
            plt.savefig('tube.png')
            plt.tight_layout()
            plt.show()

    
