# -*- coding: utf-8 -*-
'''
Reconstruct ne profile from Itot HIBP-II (TJ-II)
'''
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import integrate, signal, optimize, interpolate
from import_TS_profile import TeFit, NeShapeFunc, NeFit, ImportTS
import time
import hibplib as hb
import hibpplotlib as hbplot
import numba


# %%
# def ne_rec(rho, p1, p2, p3):
#     '''
#     ne fit using chord-averaged value neAvg
#     '''
#     coeffs = [p1, p2, p3]
#     r = np.arange(-1, 1.01, 0.01)
#     k = neAvg/(0.5*integrate.simps(NeShapeFunc(r, *coeffs), r))
#     return k*NeShapeFunc(rho, *coeffs)


def lam(r):
    return 0.004


def integrate_traj(tr, lam, get_rho, ne, coeffs, Te,
                   sigmaEff12, sigmaEff13, sigmaEff23):
    '''
    ne0, Te0 - central values
    sigmaV - interpolant over Te
    returns ne(rho)*(sigma_V(rho)/v0)*lam*exp(-integral_prim-integral_sec)
    '''
    # # first of all add the first point of secondary traj to the primary traj
    # # find the distances
    # distances = np.array([np.linalg.norm(tr.RV_prim[i, :3] - tr.RV_sec[0, :3])
    #                       for i in range(tr.RV_prim.shape[0])])
    # sorted_indices = np.argsort(distances)
    # # find position where to insert the new point
    # index_to_insert = max(sorted_indices[0:2])
    # tr.RV_prim = np.insert(tr.RV_prim, index_to_insert,
    #                        tr.RV_sec[0, :], axis=0)
    distances = np.array([np.linalg.norm(tr.RV_prim[i, :3] - tr.RV_sec[0, :3])
                          for i in range(tr.RV_prim.shape[0])])
    index_stop = np.argwhere(distances < 1e-5)[0, 0]

    # integrals over primary and secondary trajectory
    I1, I2, L1, L2, dl = 0., 0., 0., 0., 0.
    # integration loop
    # integrating primary trajectory
    for i in range(1, index_stop+1):
        r1 = tr.RV_prim[i-1, :3]
        r2 = tr.RV_prim[i, :3]

        rho1 = get_rho(r1)[0]
        rho2 = get_rho(r2)[0]

        if (rho1 <= 1.0) and (rho2 <= 1.0):
            dl = np.linalg.norm(r1 - r2)
            r_loc = (rho1 + rho2) / 2
            ne_loc = 1e19 * ne((r_loc, neAvg), *coeffs)
            Te_loc = Te(r_loc)
            I1 += dl * (sigmaEff12(Te_loc) + sigmaEff13(Te_loc)) * ne_loc
            L1 += dl

    # integrating secondary trajectory
    for i in range(1, tr.RV_sec.shape[0]):
        r1 = tr.RV_sec[i-1, :3]
        r2 = tr.RV_sec[i, :3]

        rho1 = get_rho(r1)[0]
        rho2 = get_rho(r2)[0]

        if (rho1 <= 1.0) and (rho2 <= 1.0):
            dl = np.linalg.norm(r1 - r2)
            r_loc = (rho1 + rho2) / 2
            ne_loc = 1e19 * ne((r_loc, neAvg), *coeffs)
            Te_loc = Te(r_loc)
            I2 += dl * sigmaEff23(Te_loc) * ne_loc
            L2 += dl

    r_loc = get_rho(tr.RV_sec[0, :3])[0]
    if r_loc <= 0.99:
        Te_loc = Te(r_loc)
        ne_loc = 1e19 * ne((r_loc, neAvg), *coeffs)
        sigmaEff_loc = (sigmaEff12(Te_loc) + sigmaEff13(Te_loc))
    else:
        # simple assumption for SOL
        Te_loc = Te(1.0)  # 0.
        ne_loc = 1e19 * 1e-2  # 0.
        sigmaEff_loc = (sigmaEff12(Te_loc) + sigmaEff13(Te_loc))  # 0.

    # calculate total value with integrals
    # SV size, ion zones should be calculated!
    # lam = np.linalg.norm(tr.ion_zones[2][0] - tr.ion_zones[2][-1])  # [m]
    # Itot relative to I0
    Itot = 2 * ne_loc * sigmaEff_loc * lam(r_loc) * math.exp(-I1-I2)

    return np.array([tr.Ebeam, tr.U['A2'], r_loc, Itot, ne_loc, Te_loc,
                     lam(r_loc), sigmaEff_loc, I1, I2, L1, L2])


@numba.njit()
def rho_sign(rho, Ua2):
    '''
    funtion sets rho<0 at HFS
    '''
    n = rho.shape[0]
    # set rho<0 at HFS
    distances = rho[:-1] - rho[1:]
    mask = np.argwhere(distances < -1/n)
    if mask.shape[0] == 0:
        index = distances.shape[0]
    else:
        index = mask[0][0]
    rho[:index+1] = rho[:index+1] * np.sign(Ua2[0])
    rho[index+1:] = rho[index+1:] * np.sign(Ua2[-1])
    return rho


# %%
def integrate_scan(coeffs, Iinj, traj_list, lam, get_rho, ne, Te,
                   sigmaEff12, sigmaEff13, sigmaEff23):
    ''' function calculates beam on the detector during scan
    assuming ne distribution with coeffs
    '''
    # array to contain:
    # [0]Ebeam, [1]Ua2, [2]r_loc, [3]I, [4]ne, [5]Te, [6]lam, [7]sigmaEff,
    # [8]I1, [9]I2, [10]L1, [11]L2
    Ntraj = len(traj_list)
    Ibeam = np.zeros([Ntraj, 12])

    # loop for trajectories
    for i in range(Ntraj):
        tr = traj_list[i]
        Ibeam[i, :] = integrate_traj(tr, lam, get_rho, ne, coeffs, Te,
                                     sigmaEff12, sigmaEff13, sigmaEff23)

    Ibeam[:, 3] = Iinj*Ibeam[:, 3]
    Ibeam[:, 2] = rho_sign(Ibeam[:, 2], Ibeam[:, 1])

    return Ibeam


# %%
def discrepancy(coeffs, Iinj, traj_list, lam, get_rho, ne,
                Itot_interp, Te_interp,
                sigmaEff12_interp, sigmaEff13_interp,
                sigmaEff23_interp, integrate_scan, discr_type=0):
    ''' function reprents the difference between calculated current
    and experimental values
    '''
    print('discr type ', discr_type)
    I_calculated = integrate_scan(coeffs, Iinj, traj_list, lam, get_rho,
                                  ne, Te_interp,
                                  sigmaEff12_interp, sigmaEff13_interp,
                                  sigmaEff23_interp)
    I_exp = Itot_interp(I_calculated[:, 1])
    if discr_type == 0:
        discr = np.sum((I_exp - I_calculated[:, 3])**2)*1e17
    elif discr_type == 1:
        discr = np.sum((I_exp/max(I_exp) - I_calculated[:, 3] /
                        max(I_calculated[:, 3]))**2)*1000  # *1e17
    elif discr_type == 2:
        discr = np.mean((I_exp - I_calculated[:, 3])**2)*1e17
    elif discr_type == 3:
        discr = np.sum((np.gradient(I_exp, I_calculated[:, 3])/I_exp -
                        np.gradient(I_calculated[:, 1],
                                    I_calculated[:, 3])/I_calculated[:, 1])**2)

    print('\n {}'.format(coeffs))
    print('discr = {:.6f}'.format(discr))
    return discr


def discrepancy_I0(Iinj, coeffs, traj_list, lam, get_rho, ne,
                   Itot_interp, Te_interp,
                   sigmaEff12_interp, sigmaEff13_interp,
                   sigmaEff23_interp, integrate_scan, discr_type=0):
    ''' function reprents the difference between calculated current
    and experimental values
    '''
    print('discr type ', discr_type)
    I_calculated = integrate_scan(coeffs, Iinj*1e-6, traj_list, lam, get_rho,
                                  ne, Te_interp,
                                  sigmaEff12_interp, sigmaEff13_interp,
                                  sigmaEff23_interp)
    I_exp = Itot_interp(I_calculated[:, 1])
    if discr_type == 0:
        discr = np.sum((I_exp - I_calculated[:, 3])**2)*1e17
    elif discr_type == 1:
        discr = np.sum((I_exp/max(I_exp) - I_calculated[:, 3] /
                        max(I_calculated[:, 3]))**2)*10000  # *1e17
    elif discr_type == 2:
        discr = np.mean((I_exp - I_calculated[:, 3])**2)*1e17
    elif discr_type == 3:
        discr = np.sum((np.gradient(I_exp, I_calculated[:, 3])/I_exp -
                        np.gradient(I_calculated[:, 1],
                                    I_calculated[:, 3])/I_calculated[:, 1])**2)

    print('\n {}'.format(coeffs))
    print('discr = {:.6f}'.format(discr))
    return discr


# %%
def ne_NoAtten(Iinj, Itot_interp, traj_list, lam, get_rho, Te,
               NeShapeFunc, NeFit, neAvg, sigmaEff12):
    '''functions returns Ne(rho) and coeffsNe for ne=Itot/(2*Iinj*sigma*lam)
    '''
    ne = np.zeros([0, 4])  # [0]Ebeam [1]ne [2]A2 [3]rho
    # loop for trajectories
    for i in range(len(traj_list)):
        tr = traj_list[i]
        ne_SV = 0
        r_loc = 0
        Te_loc = 0
        Ua2 = tr.U['A2']
        r_loc = get_rho([tr.RV_sec[0, 0], tr.RV_sec[0, 1], tr.RV_sec[0, 2]])[0]
        if abs(r_loc) < 0.8:
            Te_loc = Te(r_loc)
            sigmaEff_loc = sigmaEff12(Te_loc)
#            print('n_fils={}, Te={:.2f}, r={:.2f}, sigma={}'.format(n_fils,
#                                          float(Te_loc), r_loc, sigmaEff_loc))
            ne_SV = Itot_interp(Ua2)/(2*Iinj*sigmaEff_loc*lam(r_loc))
#            ne[i_No - N_start, :] = [i_No, ne_SV*1e-19, Ua2, r_loc]
            ne = np.vstack([ne, [tr.Ebeam, ne_SV*1e-19, Ua2, r_loc]])

    poptNe, pcovNe = optimize.curve_fit(NeShapeFunc, ne[:, 3], ne[:, 1],
                                        p0=[1, 0.1, -0.1], maxfev=5000)
    # normalize ne
    rho = np.arange(-1.0, 1.01, 0.01)
    k = neAvg/(0.5*integrate.simps(NeShapeFunc(rho, *poptNe), rho))
    ne[:, 1] = k*ne[:, 1]
    # make fitting of a normalized ne
    poptNe, pcovNe = optimize.curve_fit(NeFit, (ne[:, 3],
                                                np.full_like(ne[:, 3], neAvg)),
                                        ne[:, 1],
                                        p0=[1, 0.1, -0.1], maxfev=5000)

    plt.figure()
    plt.grid()
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\ n_e  (x10^{19} m^{-3})$')
    plt.axis([-1, 1, 0, 1.1*max(ne[:, 1])])
    plt.plot(ne[:, 3], ne[:, 1], '-o', color='k', label='NoAtten')
    plt.plot(rho, NeFit((rho, np.full_like(rho, neAvg)), *poptNe),
             '--', color='r', label='fit')
    plt.legend()

    return ne, poptNe


# %%
def fMaxwell(v, T, m):
    ''' Maxwelian distribution
    v in [m/s]
    T in [eV]
    '''
    if T < 0.01:
        return 0
    else:
        return ((m / (2 * np.pi * T * 1.6e-19))**1.5) * 4 * np.pi * v * v * \
            np.exp(-m * v * v / (2 * T * 1.6e-19))  # T in [eV]


def genMaxwell(vtarget, Ttarget, m_target, vbeam, m_beam):
    ''' generalized Maxwellian distribution
            Ttarget in [eV]
    '''
    Ttarget = Ttarget*1.6e-19  # go to [J]
#    v = abs(vtarget-vbeam)
    v = vbeam-vtarget
    M = m_target*m_beam/(m_beam + m_target)
    return ((M/(2*np.pi*Ttarget))**0.5) * \
        (np.exp(-M*((v-vbeam)**2)/(2*Ttarget)) -
         np.exp(-M*((v+vbeam)**2)/(2*Ttarget))) * (v/vbeam)


def dSigmaEff(vtarget, Ttarget, m_target, sigma, vbeam, m_beam):
    ''' function calculates d(effective cross section) for monoenergetic
        beam and target gas
        Ttarget in [eV]
        sigma is a function of T in [eV]
    '''
    v = abs(vtarget-vbeam)
    try:
        sigmaEff = genMaxwell(vtarget, Ttarget, m_target, vbeam, m_beam) * \
             v * sigma((0.5*m_target*v**2)/1.6e-19)
    except ValueError:
        sigmaEff = genMaxwell(vtarget, Ttarget, m_target, vbeam, m_beam) * \
             v * 0.0
    return sigmaEff


# %%
plt.close('all')

# %%
kB = 1.38064852e-23  # Boltzman [J/K]
m_e = 9.10938356e-31  # electron mass [kg]
m_p = 1.6726219e-27  # proton mass [kg]
m_ion = 133*1.6605e-27  # Cs mass [kg]
E = 132.0*1.602176634E-16  # beam energy [J]
# E = tr[0].E*1.6e-16  # beam energy [J]
v0 = math.sqrt(2*E/m_ion)  # initial particle velocity [m/s]
rho = np.arange(-1, 1.01, 0.01)

# %% LOAD IONIZATION RATES

# <sigma*v> for Cs+ + e -> Cs2+ from Shevelko
filename = 'D:\\Philipp\\Cross_sections\\Cs\\rateCs+_e_Cs2+.txt'
sigmaV12_e = np.loadtxt(filename)  # [0] Te [eV] [1] <sigma*v> [cm^3/s]
sigmaV12_e[:, 1] = sigmaV12_e[:, 1]*1e-6  # <sigma*v> goes to [m^3/s]

# <sigma*v> for Cs+ + e -> Cs3+ from Shevelko
filename = 'D:\\Philipp\\Cross_sections\\Cs\\rateCs+_e_Cs3+.txt'
sigmaV13_e = np.loadtxt(filename)  # [0] Te [eV] [1] <sigma*v> [cm^3/s]
sigmaV13_e[:, 1] = sigmaV13_e[:, 1]*1e-6  # <sigma*v> goes to [m^3/s]

# <sigma*v> for Cs+ + p -> Cs2+ from Shevelko
filename = 'D:\\Philipp\\Cross_sections\\Cs\\rateCs+_p_Cs2+.txt'
sigmaV12_p = np.loadtxt(filename)  # [0] Te [eV] [1] <sigma*v> [cm^3/s]
sigmaV12_p[:, 1] = sigmaV12_p[:, 1]*1e-6  # <sigma*v> goes to [m^3/s]

# <sigma*v> for Cs2+ + e -> Cs3+ from Shevelko
filename = 'D:\\Philipp\\Cross_sections\\Cs\\rateCs2+_e_Cs3+.txt'
sigmaV23_e = np.loadtxt(filename)  # [0] Te [eV] [1] <sigma*v> [cm^3/s]
sigmaV23_e[:, 1] = sigmaV23_e[:, 1]*1e-6  # <sigma*v> goes to [m^3/s]

# <sigma*v> for Cs2+ + p -> Cs3+ from Shevelko
filename = 'D:\\Philipp\\Cross_sections\\Cs\\rateCs2+_p_Cs3+.txt'
sigmaV23_p = np.loadtxt(filename)  # [0] Te [eV] [1] <sigma*v> [cm^3/s]
sigmaV23_p[:, 1] = sigmaV23_p[:, 1]*1e-6  # <sigma*v> goes to [m^3/s]

# %% interpolate rates
interp_type = 'quadratic'
# interp_type = 'slinear'
sigmaEff12_e_interp = interpolate.interp1d(sigmaV12_e[:, 0]/1e3,
                                           sigmaV12_e[:, 1]/v0,
                                           kind=interp_type)  # Te in [keV]
sigmaEff12_p_interp = interpolate.interp1d(sigmaV12_p[:, 0]/1e3,
                                           sigmaV12_p[:, 1]/v0,
                                           kind=interp_type)  # Te in [keV]
sigmaEff13_e_interp = interpolate.interp1d(sigmaV13_e[:, 0]/1e3,
                                           sigmaV13_e[:, 1]/v0,
                                           kind=interp_type)  # Te in [keV]
sigmaEff23_e_interp = interpolate.interp1d(sigmaV23_e[:, 0]/1e3,
                                           sigmaV23_e[:, 1]/v0,
                                           kind=interp_type)  # Te in [keV]
sigmaEff23_p_interp = interpolate.interp1d(sigmaV23_p[:, 0]/1e3,
                                           sigmaV23_p[:, 1]/v0,
                                           kind=interp_type)  # Te in [keV]

plt.figure()
plt.semilogx(sigmaV12_e[:, 0], sigmaV12_e[:, 1]*1e6, 'o', color='k',
             label=r'$Cs^+$+e $\rightarrow$ $Cs^{2+}$+2e')
Temp = np.linspace(min(sigmaV12_e[:, 0]), max(sigmaV12_e[:, 0]), num=5000)
plt.semilogx(Temp, sigmaEff12_e_interp(Temp/1e3)*1e6*v0, '-', color='k')

plt.semilogx(sigmaV13_e[:, 0], sigmaV13_e[:, 1]*1e6, 'o', color='g',
             label=r'$Cs^+$+e $\rightarrow$ $Cs^{3+}$+3e')
Temp = np.linspace(min(sigmaV13_e[:, 0]), max(sigmaV13_e[:, 0]), num=5000)
plt.semilogx(Temp, sigmaEff13_e_interp(Temp/1e3)*1e6*v0, '-', color='g')

plt.semilogx(sigmaV12_p[:, 0], sigmaV12_p[:, 1]*1e6, 'o', color='r',
             label=r'$Cs^+$+p $\rightarrow$ $Cs^{2+}$+p+e')
Temp = np.linspace(min(sigmaV12_p[:, 0]), max(sigmaV12_p[:, 0]), num=5000)
plt.semilogx(Temp, sigmaEff12_p_interp(Temp/1e3)*1e6*v0, '-', color='r')

plt.semilogx(sigmaV23_e[:, 0], sigmaV23_e[:, 1]*1e6, '^', color='k',
             label=r'$Cs^{2+}$+e $\rightarrow$ $Cs^{3+}$+2e')
Temp = np.linspace(min(sigmaV23_e[:, 0]), max(sigmaV23_e[:, 0]), num=5000)
plt.semilogx(Temp, sigmaEff23_e_interp(Temp/1e3)*1e6*v0, '--', color='k')

plt.semilogx(sigmaV23_p[:, 0], sigmaV23_p[:, 1]*1e6, 's', color='r',
             label=r'$Cs^{2+}$+p $\rightarrow$ $Cs^{3+}$+p+e')
Temp = np.linspace(min(sigmaV23_p[:, 0]), max(sigmaV23_p[:, 0]), num=5000)
plt.semilogx(Temp, sigmaEff23_p_interp(Temp/1e3)*1e6*v0, '--', color='r')

plt.xlabel(r'$E_{e,p}$ (eV)')
plt.ylabel(r'<$\sigma$V> ($cm^3/s$)')
plt.grid(linestyle='--', which='both')
leg = plt.legend()
for artist, text in zip(leg.legendHandles, leg.get_texts()):
    col = artist.get_color()
    if isinstance(col, np.ndarray):
        col = col[0]
    text.set_color(col)

plt.show()

# %% import trajectories
fname = 'E132-132_UA2-7-3_alpha74.1_beta-11.7_x270y-45z-17.pkl'
# fname = 'E144-144_UA2-8-2_alpha74.1_beta-11.7_x270y-45z-17.pkl'
traj_list = hb.read_traj_list(fname, dirname='output//100_44_64')

print('list of trajectories loaded ' + fname)

# lam_interp = interpolate.interp1d(tr.rho_ion,
#                                   signal.savgol_filter(tr.lam, 5, 3))  # [m]

# %% import experimental Itot(rho)
# rho_Ua2 should be generated in import_ro_config.py
shot = 48431  # 48435  # 47152 #44162 #44543 #47152 #48431 #44584
# file should contain t, Itot, A2, rho, Densidad2_
filename = 'D:\\Philipp\\TJ-II_programs\\Itot\\' + str(shot) + '.dat'  # + '_ne04.dat'
# filename = 'D:\\Philipp\\TJ-II_programs\\Itot\\' + str(shot) + '_ne04.dat'
Itot = np.loadtxt(filename)
# [0] time, [1] Itot, [2] Alpha2, [3] rho, [4] Densidad2_
Itot = np.delete(Itot, [2, 4, 6], axis=1)
print('experimental Itot loaded ' + filename)

neAvg = np.mean(Itot[:, 4])  # line averagend ne [e19 m-3]
neAvg = 0.77  # 0.46  # 3.6
print('\n****** ne = {:.2f}\n'.format(neAvg))

Iinj = 45e-6  #52e-6  # 100e-6  # injection beam current [A]
kAmpl = 1 * 1e7  # amplification coefficient

# flag to optimize I0
optimizeI0 = False

# flag to optimize ne
optimizeNe = False

# flag to plot parameters vs rho
addPlots = False

# add zero boundaries
# Itot_exp = np.vstack(([[1.2*np.sign(Itot[0, 3]), 0]], Itot[:, [3, 1]]))
# Itot_exp = np.vstack((Itot_exp, [1.2*np.sign(Itot_exp[-1, 0]), 0]))
# add scaling coeff
# Itot_exp[1:-1, 0] = Itot_exp[1:-1, 0]/1.15
# Itot_interp = interpolate.interp1d(Itot_exp[:, 0], Itot_exp[:, 1]/kAmpl)

Itot_interp = interpolate.interp1d(Itot[:, 2], Itot[:, 1]/kAmpl)

# %% import Thomson scattering data
shotTS = 48441  #48435  # 48431  #45784  # 49867 # 48428 #47152 #47152
t_TS = 1250  # 1270  # 1225 #1250  # 1150  # 1250
Te, TeErr, coeffsTe, Ne, NeErr, coeffsNe = ImportTS(shotTS, t_TS, neAvg,
                                                    TeFit, NeFit)
# coeffsNe = np.array([141.4147757, 5.4696319, -36.97264695])
# coeffsNe = np.array([ 3.24279103,  0.23330446, -0.97839288])

Te_interp = interpolate.interp1d(rho, TeFit(rho, *coeffsTe))

# ne_from_Itot, coeffsNe = ne_NoAtten(Iinj, Itot_interp, tr, lam_interp,
#                                     Te_interp, NeShapeFunc, NeFit, neAvg,
#                                     sigmaEff12_interp_e)

# %%
# find the best fitting Iinj
if optimizeI0:
    print('start otimization Iinj')
    t1 = time.time()
    Iinj = Iinj*1e6
    result_I0 = optimize.minimize(discrepancy_I0, Iinj,
                                  args=(coeffsNe, traj_list, lam, rho_interp,
                                        NeFit,
                                        Itot_interp, Te_interp,
                                        sigmaEff12_e_interp,
                                        sigmaEff13_e_interp,
                                        sigmaEff23_e_interp,
                                        integrate_scan, 0),
                                  method='BFGS',
                                  options={'maxiter': 1000, 'disp': True})
    t2 = time.time()
    print('Optimization time: {:.2f} min'.format((t2-t1)/60))

    Iinj = result_I0.x[0]*1e-6

# %%
if optimizeNe:
    print('start optimization ne(rho)')
    print('\n starting coeffs: \n{}'.format(coeffsNe))

    t1 = time.time()
    result = optimize.minimize(discrepancy, coeffsNe,
                               args=(Iinj, traj_list, lam, rho_interp, NeFit,
                                     Itot_interp, Te_interp,
                                     sigmaEff12_e_interp, sigmaEff13_e_interp,
                                     sigmaEff23_e_interp, integrate_scan, 1),
                               method='BFGS', tol=1e-4,
                               options={'maxiter': 1000, 'disp': True})
    t2 = time.time()
    print('Optimization time: {:.2f} min'.format((t2-t1)/60))

    coeffs_res = result.x

else:
    coeffs_res = coeffsNe

# %% plot ne and Itot
I_calculated = integrate_scan(coeffsNe, Iinj, traj_list, lam, rho_interp,
                              NeFit, Te_interp, sigmaEff12_e_interp,
                              sigmaEff13_e_interp, sigmaEff23_e_interp)

# %%
plt.rcParams.update({'font.size': 16})

# plot starting and obtained density profile
plt.figure()
plt.grid()
ne_start = [NeFit((i, neAvg), *coeffsNe) for i in rho]
plt.plot(rho, ne_start, color='b',
         label=r'starting $n_e$ profile, $\bar n_e$={:.2f}'.format(np.mean(ne_start)))

ne_result = np.array([NeFit((i, neAvg), *coeffs_res) for i in rho])
plt.plot(rho, ne_result, color='r',
         label=r'resulting $n_e$ profile, $\bar n_e$={:.2f}'.format(np.mean(ne_result)))

plt.xlim(-1.0, 1.0)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$n_e (x10^{19} m^{-3})$')
plt.title('#' + str(shot))
plt.legend()

# %%
# plot Itot profiles
plt.figure()
plt.grid()
plt.plot(I_calculated[:, 2], I_calculated[:, 3], '--o',
         color='b', label='initial iteration')
plt.errorbar(I_calculated[:, 2], Itot_interp(I_calculated[:, 1]),
             xerr=np.full_like(I_calculated[:, 2], 0.1),
             yerr=np.full_like(Itot_interp(I_calculated[:, 1]), 0.5e-7),
             capsize=3, fmt='o', color='k', label='experiment')

I_calculated = integrate_scan(coeffs_res, Iinj, traj_list, lam, rho_interp,
                              NeFit, Te_interp, sigmaEff12_e_interp,
                              sigmaEff13_e_interp, sigmaEff23_e_interp)

mask = I_calculated[:, 3] < 0
I_calculated[:, 3][mask] = 0.0

plt.plot(I_calculated[:, 2], I_calculated[:, 3], '-o', color='r',
         label='final iteration')
plt.xlim(-1.0, 1.0)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$I_{tot}$ (A)')
plt.legend()
plt.title('#' + str(shot) + r', $I_0$={:.0f} $\mu$A'.format(Iinj*1e6) +
          r', $\bar n_e$={:.2f}'.format(neAvg) +
          '\n t={}-{} ms'.format(int(min(Itot[:, 0])), int(max(Itot[:, 0]))))

# plot TS and reconstructed ne
plt.figure()
plt.errorbar(Ne[:, 0], Ne[:, 1], yerr=NeErr[:, 1], capsize=5, fmt='o')
# plt.plot(rho, NeFit((rho,np.full_like(rho, neAvg)), *coeffsNe),
#            '--', color='b', label='TS+fit')
plt.plot(rho, ne_result, color='r', label=r'reconstructed $n_e$')

plt.xlim(-1.0, 1.0)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$n_e (x10^{19} m^{-3})$')
plt.grid()
plt.title('#' + str(shotTS) + r', $\bar n_e$={:.2f}'.format(neAvg))
plt.legend()

# %%
# plot NORMALIZED Itot profiles
I_calculated = integrate_scan(coeffsNe, Iinj, traj_list, lam, rho_interp,
                              NeFit, Te_interp, sigmaEff12_e_interp,
                              sigmaEff13_e_interp, sigmaEff23_e_interp)

plt.figure()
plt.grid()
plt.plot(I_calculated[:, 2], I_calculated[:, 3]/max(I_calculated[:, 3]), '--o',
         color='b', label='initial iteration')
plt.errorbar(I_calculated[:, 2],
             Itot_interp(I_calculated[:, 1]) / max(Itot_interp(I_calculated[:, 1])),
             xerr=np.full_like(I_calculated[:, 2], 0.1),
             yerr=np.full_like(Itot_interp(I_calculated[:, 1]), 0.1),
             capsize=3, fmt='o', color='k', label='experiment')

I_calculated = integrate_scan(coeffs_res, Iinj, traj_list, lam, rho_interp,
                              NeFit, Te_interp, sigmaEff12_e_interp,
                              sigmaEff13_e_interp, sigmaEff23_e_interp)


mask = I_calculated[:, 3] < 0
I_calculated[:, 3][mask] = 0.0

plt.plot(I_calculated[:, 2], I_calculated[:, 3]/max(I_calculated[:, 3]), '-o',
         color='r', label='final iteration')
plt.xlim(-1.0, 1.0)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$Normalized I_{tot}$ (A)')
plt.legend()
plt.title('#' + str(shot) + r', $I_0$={:.0f} $\mu$A'.format(Iinj*1e6) +
          r', $\bar n_e$={:.2f}'.format(neAvg) +
          '\n t={}-{} ms'.format(int(min(Itot[:, 0])), int(max(Itot[:, 0]))))

# %%plot TS and reconstructed ne
plt.figure()
plt.errorbar(Ne[:, 0], Ne[:, 1], yerr=NeErr[:, 1], capsize=5, fmt='o',
             label='TS data')
# plt.plot(rho, NeFit((rho,np.full_like(rho, neAvg)), *coeffsNe), '--',
#          color='b', label='TS+fit')
plt.plot(rho, ne_result, color='r', label=r'reconstructed $n_e$')

plt.xlim(-1.0, 1.0)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$n_e (x10^{19} m^{-3})$')
plt.grid()
plt.title('#' + str(shotTS) + r', $\bar n_e$={:.2f}'.format(neAvg))
plt.legend()

# %%
# plot TS with new coeffs
maskTS = abs(Ne[:, 0]) < 0.05
kTS = NeFit((0.0, neAvg), *coeffs_res)/np.mean(Ne[maskTS][:, 1])

plt.figure()
plt.errorbar(Ne[:, 0], kTS*Ne[:, 1], yerr=NeErr[:, 1], capsize=5, fmt='o',
             label='TS data')
plt.plot(rho, ne_result, color='r', label=r'reconstructed $n_e$')

plt.xlim(-1.0, 1.0)
plt.legend()
plt.grid()
plt.xlabel(r'$\rho$')
plt.ylabel(r'$n_e (x10^{19} m^{-3})$')
plt.title('#' + str(shot) + r', $\bar n_e$={:.2f}'.format(neAvg))

# %%
if addPlots:
    # plot sigmaEff and atten coeffs
    mask = I_calculated[:, 2] > 1.0
    I_calculated[:, 2][mask] = 1.0
    mask = I_calculated[:, 2] < -1.0
    I_calculated[:, 2][mask] = -1.0
    mask = abs(I_calculated[:, 2]) <= 1.0

    plt.figure()
    plt.plot(I_calculated[mask][:, 2], I_calculated[mask][:, 7]*1e6, '-o')
    plt.xlim(-1.0, 1.0)
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\sigma_{eff} (cm^2)$')
    plt.grid()

    plt.figure()
    plt.plot(I_calculated[mask][:, 2], I_calculated[mask][:, 6]*1000, '-o')
    plt.xlim(-1.0, 1.0)
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\lambda_{SV}$ (mm)')
    plt.grid()

    plt.figure()
    # plt.plot(I_calculated[mask][:, 2], np.exp(-1*I_calculated[mask][:, 8]),
    #          '-o', label='primary')
    # plt.plot(I_calculated[mask][:, 2], np.exp(-1*I_calculated[mask][:, 9]),
    #          '-o', label='secondary')
    plt.plot(I_calculated[mask][:, 2],
             np.exp(-1*I_calculated[mask][:, 8]-1*I_calculated[mask][:, 9]),
             '-o')
    plt.xlim(-1.0, 1.0)
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'Atten. factor ($e^{-R_1-R_2}$)')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(I_calculated[mask][:, 2], I_calculated[mask][:, 10]*100,
             '-o', label='L primary')
    plt.plot(I_calculated[mask][:, 2], I_calculated[mask][:, 11]*100,
             '-o', label='L secondary')
    plt.xlim(-1.0, 1.0)
    plt.xlabel(r'$\rho$')
    plt.ylabel('trajectory length (cm)')
    plt.legend()
    plt.grid()

# %% final plot

I_calculated = integrate_scan(coeffsNe, Iinj, traj_list, lam, rho_interp,
                              NeFit, Te_interp, sigmaEff12_e_interp,
                              sigmaEff13_e_interp, sigmaEff23_e_interp)

fig, axs = plt.subplots(2, 3, sharex=True)
plt.rcParams.update({'font.size': 14})
title_loc = 'center'

shot_details = '#' + str(shot) + r', $t_{TS}$' + ' = {} ms, '.format(t_TS) + \
    r'$\bar n_e$' + ' = {:.2f}'.format(neAvg) + r'$x10^{19} m^{-3}$'
# plot Te
axs[0, 0].errorbar(Te[:, 0], Te[:, 1], yerr=TeErr[:, 1], capsize=5,
                   fmt='o', label='TS data')
axs[0, 0].set_ylabel(r'$\ T_e  (keV)$')
axs[0, 0].axis([-1, 1, 0, 1.1*max(Te[:, 1])])
# plot fit
axs[0, 0].plot(rho, TeFit(rho, *coeffsTe), '--', color='r', label='fit')
axs[0, 0].set_title('(a)', loc=title_loc)

# plot Ne
axs[0, 1].errorbar(Ne[:, 0], Ne[:, 1], yerr=NeErr[:, 1], capsize=5,
                   fmt='o', label='TS data')
axs[0, 1].set_ylabel(r'$\ n_e  (x10^{19} m^{-3})$')
axs[0, 1].axis([-1, 1, 0, 1.1*max(Ne[:, 1])])
# axs[0,1].set_title(shot_details)
# plot fit
axs[0, 1].plot(rho, NeFit((rho, np.full_like(rho, neAvg)), *coeffsNe),
               '--', color='r', label='fit')
axs[0, 1].set_title('(b)', loc=title_loc)

# plot TS with new coeffs
ne_result = np.array([NeFit((i, neAvg), *coeffs_res) for i in rho])

maskTS = abs(Ne[:, 0]) < 0.05
kTS = NeFit((0.0, neAvg), *coeffs_res) / np.mean(Ne[maskTS][:, 1])
kTS = 1

axs[1, 1].errorbar(Ne[:, 0], kTS*Ne[:, 1], yerr=NeErr[:, 1],
                   capsize=5, fmt='o', label='TS data')
axs[1, 1].plot(rho, ne_result, color='r', label=r'reconstructed $n_e$')
axs[1, 1].set_ylabel(r'$n_e (x10^{19} m^{-3})$')
axs[1, 1].set_title('(e)', loc=title_loc)

# plot Itot profiles
axs[1, 2].plot(I_calculated[:, 2], I_calculated[:, 3], '--o',
               color='b', label='initial iteration')
axs[1, 2].errorbar(I_calculated[:, 2], Itot_interp(I_calculated[:, 1]),
                   xerr=np.full_like(I_calculated[:, 2], 0.1),
                   yerr=np.full_like(Itot_interp(I_calculated[:, 1]), 0.5e-7),
                   capsize=3, fmt='o', color='k', label='experiment')

I_calculated = integrate_scan(coeffs_res, Iinj, traj_list, lam, rho_interp,
                              NeFit, Te_interp, sigmaEff12_e_interp,
                              sigmaEff13_e_interp, sigmaEff23_e_interp)

mask = I_calculated[:, 3] < 0
I_calculated[:, 3][mask] = 0.0

axs[1, 2].plot(I_calculated[:, 2], I_calculated[:, 3], '-o',
               color='r', label='final iteration')
axs[1, 2].set_ylabel(r'$I_{tot}$ (A)')
axs[1, 2].set_title('(f)', loc=title_loc)

# plot atten factor
mask = I_calculated[:, 2] > 1.0
I_calculated[:, 2][mask] = 1.0
mask = I_calculated[:, 2] < -1.0
I_calculated[:, 2][mask] = -1.0
mask = abs(I_calculated[:, 2]) <= 1.0
axs[0, 2].plot(I_calculated[mask][:, 2],
               np.exp(-1*I_calculated[mask][:, 8]-1*I_calculated[mask][:, 9]),
               '-o')
axs[0, 2].set_ylabel(r'Atten. factor ($e^{-R_1-R_2}$)')
axs[0, 2].set_title('(c)', loc=title_loc)
# plot sigmaEff
axs[1, 0].plot(I_calculated[mask][:, 2], I_calculated[mask][:, 7]*1e4, '-o')
axs[1, 0].set_ylabel(r'$\sigma_{eff} (cm^2)$')
axs[1, 0].set_title('(d)', loc=title_loc)

# format axes
for ax in fig.get_axes():
    ax.set_xlabel(r'$\rho$')
    ax.set_xlim(-1.0, 1.0)
    ax.grid()
    ax.legend()

# fig.tight_layout()
