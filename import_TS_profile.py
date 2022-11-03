# -*- coding: utf-8 -*-
"""
Import TS profile for TJ-II stellarator
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize, integrate
from bessfit import bessfit2, bessfit3


# %% ITERFEROMETER CHORD
INTERF_CHORD = np.loadtxt('tj2lib//interf_rzfi_rho.txt')  # R, fi, Z, rho
L_CHORD = np.linalg.norm(INTERF_CHORD[:, [0, 2]] - INTERF_CHORD[0, [0, 2]],
                         axis=1)

# %% TS data download
def NeShapeFunc(rho, p1, p2, p3, p4):
    ''' function to approximate ne shape
    by van Milligen
    '''
    # return p1*(np.exp(-p2*(1 - rho**2) - p3*(1 - rho**4)) - 1)
    # return p1*np.exp(-p2*(1 - rho**2) - p3*(1 - rho**4))
    # return p1*np.exp(-p2*rho**2 - p3*rho**4 - p4*rho**6)
    # return p1*(1 - rho**(2*abs(p2)))**abs(p3)
    # return (p1*(1 - rho**2) + p2*(1 - rho**4))*np.exp(-p3*rho**2)
    # return p1*(np.exp(-p2*(1 - rho**2) - p3*(1 - rho**4)))
    return p1*np.exp(-p2*rho**2 - p3*rho**4 - p4*rho**6)
    # return bessfit3(rho, p1, p2, p3)

def normalize_Ne(coeffsNe, neAvg):
    ''' function calculates normalization coefficient 
    so that line averaged value was equal to neAvg
    '''
    r = INTERF_CHORD[:, 3]
    L = L_CHORD[-1]  # total langth of the chord in m
    return neAvg * L / integrate.simps(NeShapeFunc(r, *coeffsNe), L_CHORD)

def NeFit(param, p1, p2, p3, p4):
    ''' ne fit using chord-averaged value
    '''
    rho, neAvg = param[0], param[1]
    coeffs_new = [p1, p2, p3, p4]
    k = normalize_Ne(coeffs_new, neAvg)
    return k*NeShapeFunc(rho, *coeffs_new)
#    return 1*(np.exp(-p2*(1 - rho**2) - p3*(1 - rho**4)) - 1)
#    return (p1*(1-rho**2) + p2*(1-rho**4))*np.exp(-p3*rho**2)


def TeFit(rho, p1, p2, p3, p4):
# def TeFit(x, a, b):
# def TeFit(x, y0, A, x_c, w, a, b):
# def TeFit(x, y0, a, w):
    ''' Te fit function
    '''
    # return y0 + a*np.exp(-(x**2) / (2*(w)**2))  # Gauss
    # return y0*(np.exp(-a*(1-x**2) - w*(1-x**4)) - 1)
    # return y0 + a*np.exp(-0.5*((x-xc)/w)**2)
    return p1*np.exp(-p2*rho**2 - p3*rho**4 - p4*rho**6)

    # p = 1
    # return a*((1 + b*np.abs(x)**p)*np.exp(-b*np.abs(x)**p) - (1 + b)*np.exp(-b))

    # z = (x - x_c)/w
    # return y0 + A*np.exp(-0.5 * z**2) / (w*np.sqrt(2*np.pi)) * (1 + np.abs( (a/6)*(z**3 - 3*z) + (b/24)*(z**4 - 6*z**3 + 3) ))

    # p1, p2, p3, p4, rho = y0, a, b, w, x
    # return (p1*(1 - rho**2) + p2*(1 - rho**4))*np.exp(-p3*rho**2)


# %%
def ImportTS(shot, t_TS, neAvg, TeFit, NeFit,
             # coeffsTe0= [1, 1, 1, 1, 1, 1],
               # coeffsTe0=[1, 1, 1],
              # coeffsTe0 = [0.05, -5, 2, 1.0],
               coeffsTe0 = [1., 1., 1., 1.],
             coeffsNe0=[1, 1, 1, 1], plot_TS=True, t_He=1201):
    ''' TeFit and NeFit are functions for Ne and Te fitting
    '''
    # import Te
    filename = 'D:\\cache\\Thomson.2\\PerfilTe_' + str(shot) + '_' + \
        str(t_TS) + '.dat'
    Te = np.loadtxt(filename)  # [0]rho [1]Te
    # import Te errors
    filename = 'D:\\cache\\Thomson.2\\PerfildTe_' + str(shot) + '_' + \
        str(t_TS) + '.dat'
    TeErr = np.loadtxt(filename)  # [0]rho [1]TeErr
    # import Ne
    filename = 'D:\\cache\\Thomson.2\\PerfilNe_' + str(shot) + '_' + \
        str(t_TS) + '.dat'
    Ne = np.loadtxt(filename)  # [0]rho [1]Ne
    # import Ne errors
    filename = 'D:\\cache\\Thomson.2\\PerfildNe_' + str(shot) + '_' + \
        str(t_TS) + '.dat'
    NeErr = np.loadtxt(filename)  # [0]rho [1]NeErr

    print('Shot ' + str(shot) + ', TS loaded')
    # calculate central Te and deviation
    mask = (Te[:, 0] > -0.05) & (Te[:, 0] < 0.05)
    Te_center = Te[mask, 1]
    Te0 = np.mean(Te_center)
    std_dev = np.std(Te_center)
    max_err = max(abs(Te_center - Te0))

    # %% import He beam data
    try:
        filename = 'D:\\cache\\Hebeam\\Te_'+ str(shot) + '_' + \
            str(t_He) + '.dat'
        Te_edge = np.loadtxt(filename)
        Te_edge[:, 1] = Te_edge[:, 1]/1000.  # go to keV
        filename = 'D:\\cache\\Hebeam\\Ne_'+ str(shot) + '_' + \
            str(t_He) + '.dat'
        Ne_edge = np.loadtxt(filename)
        print('He beam data imported, t = ', t_He)
        Hebeam_loaded = True
    except (FileNotFoundError, IOError) as e:
        # set edge values manually [[rho, Te in keV], [...]]
        print(e)
        # Te_edge = [[0.8, 0.06], [1, 0.04]]*1000
        # Te_edge = [[0.8, 0.05], [1, 0.02]]*1000
        Te_edge = [[0.8, 0.04], [1, 0.01]]*1000
        # Ne_edge = [[0.8, 0.3], [1, 0.05]]*1000
        # Ne_edge = [[0.8, 0.8], [1, 0.01]]*1000
        Ne_edge = [[0.8, 0.4], [1, 0.02]]*1000
        # Ne_edge = [[0.8, 0.3], [1, 0.02]]*1000
        Hebeam_loaded = False

    # %% make fitting of TS data
    # fitting of Te
    rho_min = -0.1
    rho_max = 0.4
    mask_Te = (Te[:, 0] > rho_min) & (Te[:, 0] < rho_max)
    Te_data = Te[mask_Te]

    # add boundary zeros
    Te_data = np.vstack((Te_data, Te_edge))

    poptTe, pcovTe = optimize.curve_fit(TeFit, Te_data[:, 0], Te_data[:, 1],
                                        p0=coeffsTe0, maxfev=5000)

    # make fitting of Ne shape
    rho_min = -0.2
    rho_max = 0.7
    mask_Ne = (Ne[:, 0] > rho_min) & (Ne[:, 0] < rho_max)
    Ne_data = Ne[mask_Ne]

    # add boundary zeros
    Ne_data = np.vstack((Ne_data, Ne_edge))

    poptNe, pcovNe = optimize.curve_fit(NeShapeFunc, Ne_data[:, 0],
                                        Ne_data[:, 1], p0=coeffsNe0,
                                        maxfev=5000)
    # normalize experimental Ne
    k = normalize_Ne(poptNe, neAvg)
    print('\n' + str(shot) + ' Ne coeff = {:.3f}\n'.format(k))
    Ne[:, 1] = k*Ne[:, 1]

    # %% plot TS data
    if plot_TS:
        # poptNe = [2.7547063 , 0.59897433, 4.14143101, 2.34357011]
        # poptTe = [0.32658426, 2.82169597, 0.79468697, 0.12544259]
        
        rho = np.arange(-1.02, 1.02, 0.01)
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
        if Hebeam_loaded:
            plt.plot(Te_edge[:, 0], Te_edge[:, 1],
                     '-o', color='g', label='He beam')
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
        if Hebeam_loaded:
            plt.plot(Ne_edge[:, 0], Ne_edge[:, 1],
                     '-o', color='g', label='He beam')
        plt.legend()

    # return Te[mask_Te], TeErr[mask_Te], poptTe, Ne[mask_Ne], NeErr[mask_Ne], poptNe
    return Te, TeErr, poptTe, Ne, NeErr, poptNe


# %%
def saveECEcalib(filename, calib):
    ''' function saves ECE calib'''
    np.savetxt(filename, calib, delimiter=' ',
               header='ECE_channel calibration_factor',
               fmt='%2.4f', newline='\r\n')
    print('SAVED FILE: ' + filename)
    return


def save_processed_TS(Te, TeErr, Ne, NeErr, shot, t_TS,
                      filepath='D:\\NRCKI\\2022\\TJII\\for_DebBasu_ICRH\\TS data\\'):
    # save Te
    filename = filepath + 'Te_' + str(shot) + '_' + str(t_TS) + '.dat'
    np.savetxt(filename, Te)
    filename = filepath + 'dTe_' + str(shot) + '_' + str(t_TS) + '.dat'
    np.savetxt(filename, TeErr)
    # save Ne
    filename = filepath + 'Ne_' + str(shot) + '_' + str(t_TS) + '.dat'
    np.savetxt(filename, Ne)
    filename = filepath + 'dNe_' + str(shot) + '_' + str(t_TS) + '.dat'
    np.savetxt(filename, NeErr)
    print('processed TS data saved\n')

    return


def save_TS_fit(coeffsTe, coeffsNe, neAvg):
    rho_array = np.arange(-1, 1.01, 0.01)
    Te_array = np.array([rho_array, TeFit(rho_array, *coeffsTe)]).T
    Ne_array = np.array([rho_array,
                         NeFit((rho_array, np.full_like(rho_array, neAvg)),
                               *coeffsNe)]).T
    filename = 'D:\\cache\\TSfit\\Te_' + str(shot) + '.dat'
    np.savetxt(filename, Te_array)
    filename = 'D:\\cache\\TSfit\\Ne_' + str(shot) + '.dat'
    np.savetxt(filename, Ne_array)
    print('TSfits SAVED')

    return


# %%
if __name__ == '__main__':

    plt.close('all')
    shot = 52659  # 52695  #50186 # 52548  # 50489  # 48441 # 48431  #50533  # 48435  # 48441  #48428  #48431 #48435 #47152 #44354 # 48441
    t_TS = 1235  # 1210  # 1205  # 1120  # 1140  # 1250
    # line-averaged density [e19 m-3]
    neAvg = 3.79      # 3.18  # 3.2 # 2.38  # 0.33  # 0.8  #0.46
    # load Thomson profiles
    Te, TeErr, coeffsTe, Ne, NeErr, coeffsNe = ImportTS(shot, t_TS, neAvg,
                                                        TeFit, NeFit)
    # save_processed_TS(Te, TeErr, Ne, NeErr, shot, t_TS)
    # save_TS_fit(coeffsTe, coeffsNe, neAvg)

    try:
        # load ECE
        filename = 'G:\\cache\\ECE_dat\\' + str(shot) + '_ECE' + '.dat'
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

    except (FileNotFoundError, IOError) as e:
        print(e)
        print(filename + ' NOT FOUND')
