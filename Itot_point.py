# -*- coding: utf-8 -*-
"""
Claculate Itot (neAvg) evolution
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import integrate, signal, optimize, interpolate
from import_TS_profile import TeFit, NeShapeFunc, NeFit, ImportTS, normalize_Ne
from reconstruct_ne import *
import time
import hibplib as hb
import hibpplotlib as hbplot
import numba


# %% find the closest to UA2_aim traj in traj_list
distances = []
UA2_aim = 1.21  # -1.9
for tr in traj_list:
    distances.append(abs(UA2_aim - tr.U['A2']))
trajectory = traj_list[np.argmin(distances)]

# %%
# load shot list
fname = 'D:\\NRCKI\\2022\\TJII\\TS_NBI_cntr.txt'
# fname = 'D:\\NRCKI\\2022\\TJII\\TS_NBI_co.txt'
# fname = 'D:\\NRCKI\\2022\\TJII\\TS_NBI.txt'
data = np.loadtxt(fname)
# sort by neAvg
data = data[np.argsort(data[:, 1])]

# create array to story Itot
Nshots = data.shape[0]
Itot = np.zeros([Nshots, 12])
# create array for neAvg Te0
neAvgTe0 = []

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
figPe, ax3 = plt.subplots()

rho = np.arange(-1.02, 1.02, 0.01)
for i in range(Nshots):
# for i in range(21, 27):
    # get TS data
    shotTS = int(data[i, 0])
    neAvg = data[i, 1]
    t_TS = int(data[i, 2])
    if neAvg < 1.5:
        coeffsTe = np.array([ 0.00814735,  0.43716948, -0.37038859])
        TeValues = np.array([TeFit(r, *coeffsTe) for r in rho])
        Te0 = 0.615 - 0.069*neAvg
        kTe = Te0/max(TeValues)
        TeValues = kTe*TeValues
        Te_interp = interpolate.interp1d(rho, TeValues)
    else:
        Te, TeErr, coeffsTe, Ne, NeErr, coeffsNe = ImportTS(shotTS, t_TS, neAvg,
                                                            TeFit, NeFit, t_He=0.0,
                                                            plot_TS=False)
        Te_interp = interpolate.interp1d(rho, TeFit(rho, *coeffsTe))

    # calculate Pe
    Pe = np.copy(Ne)
    Pe[:, 1] = Pe[:, 1] * Te[:, 1]
    poptPe, pcovPe = optimize.curve_fit(TeFit, Pe[:, 0], Pe[:, 1],
                                        p0=[1, 1, 1], maxfev=5000)
    # plot Pe
    PeFit = TeFit(rho, *coeffsTe) * NeFit((rho, np.full_like(rho, neAvg)), *coeffsNe)
    ax3.plot(rho, PeFit/max(PeFit),
             label=str(shotTS) + ', ne={:.2f}'.format(neAvg))
    # ax3.plot(rho, TeFit(rho, *poptPe),
    #          label=str(shotTS) + ', ne={:.2f}'.format(neAvg))
    # get central Te
    neAvgTe0.append([neAvg, Te_interp(0.0)])
    # plot TS data
    ax1.plot(rho, TeFit(rho, *coeffsTe),
             label=str(shotTS) + ', ne={:.2f}'.format(neAvg))
    ax2.plot(rho, NeFit((rho, np.full_like(rho, neAvg)), *coeffsNe),
             label=str(shotTS) + ', ne={:.2f}'.format(neAvg))
    # integrate trajectory
    Itot[i, :] = integrate_traj(trajectory, lam, rho_interp,
                                NeFit, neAvg, coeffsNe, Te_interp,
                                sigmaEff12_e_interp, sigmaEff13_e_interp,
                                sigmaEff23_e_interp, drho=0.01)
    Itot[:, 2] = rho_sign(Itot[:, 2], Itot[:, 1])

ax1.set_xlabel(r'$\rho$')
ax1.set_ylabel(r'$\bar T_e$ (keV)')
ax1.grid()
ax2.set_xlabel(r'$\rho$')
ax2.set_ylabel(r'$\bar n_e (x10^{19} m^{-3})$')
ax2.grid()
ax3.set_xlabel(r'$\rho$')
ax3.set_ylabel(r'$P_e$ (a.u.)')
ax3.grid()
plt.legend()

neAvgTe0 = np.array(neAvgTe0)

# %% plot Te0 vs neAvg
plt.figure()
plt.plot(neAvgTe0[:, 0], neAvgTe0[:, 1], 'o')
plt.xlabel(r'$\bar n_e$')
plt.ylabel(r'$T_{e0}$')
plt.grid()

# %% plot Itot vs neAvg
plt.figure()
plt.plot(neAvgTe0[:, 0], Itot[:, 3]/max(Itot[:, 3]), '-o')
plt.xlabel(r'$\bar n_e$')
plt.ylabel(r'$I_{tot}$ (a.u.)')
plt.grid()
