# -*- coding: utf-8 -*-
"""
Import TS profile for TJ-II stellarator
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize, integrate
from bessfit import bessfit2, bessfit3


# %% TS data download
def NeShapeFunc(rho, p1, p2, p3):
    ''' function to approximate ne shape
    by van Milligen '''
#    return p1*(np.exp(-p2*(1 - rho**2) - p3*(1 - rho**4)) - 1)
#    return p1*np.exp(-p2*rho**2 - p3*rho**4 - p4*rho**6)
#    return p1*(1 - rho**(2*abs(p2)))**abs(p3)
    # return (p1*(1 - rho**2) + p2*(1 - rho**4))*np.exp(-p3*rho**2)
    return bessfit3(rho, p1, p2, p3)


def NeFit(param, p1, p2, p3):
    '''ne fit using chord-averaged value
    '''
    rho = param[0]
    neAvg = param[1]
    coeffs_new = [p1, p2, p3]

    r = np.arange(-1, 1.01, 0.01)
    k = neAvg/(0.5*integrate.simps(NeShapeFunc(r, *coeffs_new), r))

    return k*NeShapeFunc(rho, *coeffs_new)
#    return 1*(np.exp(-p2*(1 - rho**2) - p3*(1 - rho**4)) - 1)
#    return (p1*(1-rho**2) + p2*(1-rho**4))*np.exp(-p3*rho**2)


# def TeFit(x, y0, a, xc, w):
#     ''' Gauss function
#     '''
#     return y0 + a*np.exp(-0.5*((x-xc)/w)**2)


def TeFit(x, y0, a, w):
# def TeFit(x, a, b):
# def TeFit(x, y0, A, x_c, w, a, b):
    ''' Te fit function
    '''
    return y0 + a*np.exp(-(x**2) / (2*(w)**2)) # Gauss
    # return y0*(np.exp(-a*(1-x**2) - w*(1-x**4)) - 1)

    # p = 1
    # return a*((1 + b*np.abs(x)**p)*np.exp(-b*np.abs(x)**p) - (1 + b)*np.exp(-b))

    # z = (x - x_c)/w
    # return y0 + A*np.exp(-0.5 * z**2) / (w*np.sqrt(2*np.pi)) * (1 + np.abs( (a/6)*(z**3 - 3*z) + (b/24)*(z**4 - 6*z**3 + 3) ))

#    p1, p2, p3, p4, rho = y0, a, b, w, x
#    return (p1*(1 - rho**2) + p2*(1 - rho**4))*np.exp(-p3*rho**2)


# %%
def ImportTS(shot, t_TS, neAvg, TeFit, NeFit,
             # coeffsTe0= [1, 1, 1, 1, 1, 1],
              coeffsTe0=[1, 1, 1],
             # coeffsTe0 = [0.05, -5, 2, 1.0],
             coeffsNe0=[1, 1, 1], plot_TS=True):
    ''' TeFit and NeFit are functions for Ne and Te fitting
    '''
    # import Te
    filename = 'E:\\cache\\Thomson.2\\PerfilTe_' + str(shot) + '_' + \
        str(t_TS) + '.dat'
    Te = np.loadtxt(filename)  # [0]rho [1]Te
    # import Te errors
    filename = 'E:\\cache\\Thomson.2\\PerfildTe_' + str(shot) + '_' + \
        str(t_TS) + '.dat'
    TeErr = np.loadtxt(filename)  # [0]rho [1]TeErr
    # import Ne
    filename = 'E:\\cache\\Thomson.2\\PerfilNe_' + str(shot) + '_' + \
        str(t_TS) + '.dat'
    Ne = np.loadtxt(filename)  # [0]rho [1]Ne
    # import Ne errors
    filename = 'E:\\cache\\Thomson.2\\PerfildNe_' + str(shot) + '_' + \
        str(t_TS) + '.dat'
    NeErr = np.loadtxt(filename)  # [0]rho [1]NeErr

    print('Shot ' + str(shot) + ' loaded')

    mask = (Te[:, 0] > -0.05) & (Te[:, 0] < 0.05)
    Te_center = Te[mask, 1]
    Te0 = np.mean(Te_center)
    std_dev = np.std(Te_center)
    max_err = max(abs(Te_center - Te0))

    # %% make fitting of TS data
    # fitting of Te
    rho_min = -0.6
    rho_max = 0.1
    mask_Te = (Te[:, 0] > rho_min) & (Te[:, 0] < rho_max)
    Te_data = Te[mask_Te]

#    Te = Te[mask]
#    TeErr = TeErr[mask]
#    Ne = Ne[mask]
#    NeErr = NeErr[mask]

    # add boundary zeros
    Te_data = np.vstack(([[-1, 0.04], [-0.8, 0.06]]*1000, Te_data))
    Te_data = np.vstack((Te_data, [[0.8, 0.06], [1, 0.04]]*1000))
    # Te_data = np.vstack(([[-1, 0.0]]*100, Te_data))
    # Te_data = np.vstack((Te_data, [[1, 0.0]]*100))

    poptTe, pcovTe = optimize.curve_fit(TeFit, Te_data[:, 0], Te_data[:, 1],
                                        p0=coeffsTe0, maxfev=5000)

    # make fitting of Ne shape
    rho_min = -0.5
    rho_max = 0.8
    mask_Ne = (Ne[:, 0] > rho_min) & (Ne[:, 0] < rho_max)
    Ne_data = Ne[mask_Ne]

    poptNe, pcovNe = optimize.curve_fit(NeShapeFunc, Ne_data[:, 0],
                                        Ne_data[:, 1], p0=coeffsNe0,
                                        maxfev=5000)
    # normalize Ne
    rho = np.arange(-1.0, 1.01, 0.01)
    k = neAvg/(0.5*integrate.simps(NeShapeFunc(rho, *poptNe), rho))
    print('\n' + str(shot) + ' Ne coeff = {:.3f}\n'.format(k))
    Ne[:, 1] = k*Ne[:, 1]
    Ne_data = Ne[mask_Ne]
    # make fitting of a normalized Ne
    poptNe, pcovNe = optimize.curve_fit(NeFit,
                                        (Ne_data[:, 0],
                                         np.full_like(Ne_data[:, 1], neAvg)),
                                        Ne_data[:, 1],
                                        p0=coeffsNe0, maxfev=5000)

    # %% plot TS data
    if plot_TS:
        plt.rcParams.update({'font.size': 14})
        shot_details = '#' + str(shot) + r', $t_{TS}$' + ' = {} ms, '.format(t_TS) + \
                          r'$\bar n_e$' +' = {:.2f}'.format(neAvg) + r'$x10^{19} m^{-3}$'
        # plot Te
        plt.figure()
        plt.errorbar(Te[mask_Te, 0], Te[mask_Te, 1],
                     yerr=TeErr[mask_Te, 1], capsize=5, fmt='o', label='TS data')
    #    plt.plot(Te[:,0], Te[:,1], '-o', label='TS data')
        plt.grid()
        plt.xlabel(r'$\rho$')
        plt.ylabel(r'$\ T_e  (keV)$')
        plt.axis([-1, 1, 0, 1.1*max(Te[:, 1])])
        plt.title(shot_details)
    #              '\nTe0_avg = {:0.3f} keV, '.format(Te0) + \
    #              'std_dev = {:0.3f}, '.format(std_dev) + \
    #              'max_err = {:0.3f}'.format(max_err))

        plt.plot(rho, TeFit(rho, *poptTe), '--', color='r', label='fit')
        plt.legend()

        # plot Ne
        plt.figure()
        plt.errorbar(Ne[mask_Ne, 0], Ne[mask_Ne, 1],
                     yerr=NeErr[mask_Ne, 1], capsize=5, fmt='o', label='TS data')
        plt.grid()
        plt.xlabel(r'$\rho$')
        plt.ylabel(r'$\ n_e  (x10^{19} m^{-3})$')
        plt.axis([-1, 1, 0, 1.1*max(Ne[:, 1])])
        plt.title(shot_details)

        plt.plot(rho, NeFit((rho, np.full_like(rho, neAvg)), *poptNe),
                 '--', color='r', label='fit')
        plt.legend()

    return Te, TeErr, poptTe, Ne, NeErr, poptNe


# %%
def saveECEcalib(filename, calib):
    ''' function saves ECE calib'''
    np.savetxt(filename, calib, delimiter=' ',
               header='ECE_channel calibration_factor',
               fmt='%2.4f', newline='\r\n')
    print('SAVED FILE: ' + filename)
    return


# %%
if __name__ == '__main__':

    plt.close('all')
    shot = 48441 # 48431  #50533  # 48435  # 48441  #48428  #48431 #48435 #47152 #44354 # 48441
    t_TS = 1250  #1140  # 1250 #1250 #1250
    neAvg = 0.8  #0.46  # line-averaged density [e19 m-3]

#    plt.close('all')
    # load Thomson profiles
    Te, TeErr, coeffsTe, Ne, NeErr, coeffsNe = ImportTS(shot, t_TS, neAvg, TeFit, NeFit)

#    rho_array = np.arange(-1,1.01,0.01)
#    Te_array = np.array([rho_array, TeFit(rho_array, *coeffsTe)]).T
#    Ne_array = np.array([rho_array, NeFit((rho_array,np.full_like(rho_array, neAvg)), *coeffsNe)]).T
#    filename = 'D:\\cache\\TSfit\\Te_' + str(shot) + '.dat'
#    np.savetxt(filename, Te_array)
#    filename = 'D:\\cache\\TSfit\\Ne_' + str(shot) + '.dat'
#    np.savetxt(filename, Ne_array)
#    print('TSfits SAVED')

    try:
        # load ECE
        filename = 'E:\\cache\\ECE_dat\\' + str(shot) + '_ECE' + '.dat'
        ECE = np.loadtxt(filename)  # [0]t [1]ECE1 [2]t [3]ECE2 [4]t ...
        print('ECE loaded')
        # delete time columns except [0]
        ECE = np.delete(ECE, np.arange(2, ECE.shape[1], 2), 1)
        # shift ECE to zero
        N = ECE.shape[1]
        Nz = 500
        zero_levels = np.mean(ECE[0:Nz, 1:], axis=0)
        ECE[:, 1:] = (ECE[:, 1:] - zero_levels)

        # load ECE calibration
#        filename = 'D:\\NRCKI\\TJ-II_programs\\TeGUI\\ECE_calib\\ECE_calib_' + \
#            str(shot) + '.txt'
#        calib = np.loadtxt(filename)
#        print('ECE calib loaded')

        t_ECE = t_TS
        dt = 2  # range of averaging [ms]

        rho_ECE = np.array([-0.75299,
                            -0.64574,
                            -0.53182,
                            -0.44733,
                            -0.35098,
                            -0.24543,
                            -0.18912,
                            -0.12894,
                            -0.0728,
                            -0.01242,
                            0.05248,
                            0.12194])

        # choose ECE time range
        mask = (ECE[:, 0] < t_ECE+dt) & (ECE[:, 0] > t_ECE)
        ECE_cut = np.c_[rho_ECE, np.mean(ECE[mask][:, 1:], axis=0)]
        # calibrate ECE
        calibECE = np.c_[np.arange(1,N), TeFit(ECE_cut[:,0], *coeffsTe) / ECE_cut[:, 1]]

        # plot ECE and TSfit
        plt.figure()
        plt.title('#' + str(shot) + ', t={:.2f} ms'.format(t_ECE))
        plt.plot(ECE_cut[:, 0], -1*ECE_cut[:, 1], '-o', color='g', label='raw ECE')
        plt.plot(ECE_cut[:, 0], TeFit(ECE_cut[:, 0], *coeffsTe), label='TS fit')

        # plot calibrated ECE
        plt.plot(ECE_cut[:, 0], ECE_cut[:, 1]*calibECE[:, 1], '*', color='r', label='ECE calib')
        plt.legend()
        plt.grid()
        plt.xlabel(r'$\rho$')
        plt.ylabel(r'$\ T_e  (keV)$')

        # save ECE calib
        filename = 'D:\\NRCKI\\TJ-II_programs\\TeGUI\\ECE_calib\\ECE_calib_' + \
            str(shot) + '.txt'
#        saveECEcalib(filename, calibECE)

    except FileNotFoundError:
        print(filename + ' NOT FOUND')
