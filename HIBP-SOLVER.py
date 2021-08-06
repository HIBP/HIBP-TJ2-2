'''
TJ2 stellarator, HIBP-II

Program calculates trajectories and selects voltages on
primary beamline (B2 plates) and secondary beamline (A3, B3, B4, A4 plates)
'''

import numpy as np
import hibplib as hb
import hibpplotlib as hbplot
import define_geometry as defgeom
import copy
import time

# %% set up main parameters
# choose analyzer number
analyzer = 1

# magnetic configuration
config = '100_44_64'
print('\nShot parameters: config ' + config)

# timestep [sec]
dt = 0.4e-7  # 0.7e-7

# probing ion charge and mass
q = 1.602176634e-19  # electron charge [Co]
m_ion = 132.905 * 1.6605e-27  # Cs ion mass [kg]

# beam energy
Emin, Emax, dEbeam = 148., 148., 2.

# set flags
optimizeB2 = False
optimizeA3B3 = False
pass2AN = True
save_radref = False

# A1 and B1 plates voltages
UA1, UB1 = 0.1, 0.75  # [kV]

# UA2 voltages
UA2min, UA2max, dUA2 = -9., 5., 2.
NA2_points = 6

# B2 plates voltage
UB2, dUB2 = 2.0, 35.0  # [kV], [kV/m]

# B3 voltages
UB3, dUB3 = 0.0, -25.0  # [kV], [kV/m]

# A3 voltages
UA3, dUA3 = 0.0, -25.0  # [kV], [kV/m]

# A4 voltages
UA4, dUA4 = 0.0, 2.0  # [kV], [kV/m]

# B4 voltages
UB4, dUB4 = 0.0, 2.0  # [kV], [kV/m]

# %% Define Geometry
geomTJ2 = defgeom.define_geometry(config, analyzer=analyzer)
r0 = geomTJ2.r_dict['r0']  # trajectory starting point

# angles of aim plane normal [deg]
alpha_aim = 0.
beta_aim = 0.
stop_plane_n = hb.calc_vector(1.0, alpha_aim, beta_aim,
                              direction=(1, 1, 1))

# %% Load Magnetic Field
if 'B' not in locals():
    dirname = 'tj2lib'
    B, rho_interp = hb.read_B(config, dirname=dirname, interp=True)

# %% Load Electric Field
E = {}
# load E for primary beamline
try:
    hb.read_plates('prim', geomTJ2, E)
    print('\n Primary Beamline loaded')
except FileNotFoundError:
    print('\n Primary Beamline NOT FOUND')

# load E for secondary beamline
try:
    E_sec = hb.read_plates('sec', geomTJ2, E)
    print('\n Secondary Beamline loaded')
except FileNotFoundError:
    print('\n Secondary Beamline NOT FOUND')

# %% Analyzer parameters
if 'an' in geomTJ2.plates_dict.keys():
    # Analyzer G
    G = geomTJ2.plates_dict['an'].G
    # add detector coords to dictionary
    edges = geomTJ2.plates_dict['an'].det_edges
    geomTJ2.r_dict['det'] = edges[edges.shape[0]//2][0]
else:
    G = 1.
    print('\nNO Analyzer')

# %% Optimize Primary Beamline
# define list of trajectories that hit r_aim
traj_list_B2 = []
# initial beam energy range
Ebeam_range = np.arange(Emin, Emax + dEbeam, dEbeam)  # [keV]

for Ebeam in Ebeam_range:
    t1 = time.time()
    # set up scanning voltage
    ## apr2019
    if Ebeam < 105:
        shot = '48425'
    else:
        shot = '48435'

    ## 18dec2019
    # if Ebeam < 93.:
    #     shot = '49858'
    # elif Ebeam < 101.:
    #     shot = '49861'
    # else:
    #     shot = '49878'

    ## mar2020
    # if Ebeam == 92:
    #     shot = '50561'
    # elif Ebeam == 96 or Ebeam == 100:
    #     shot = '50559'
    # elif Ebeam == 120:
    #     shot = '50553'
    # elif Ebeam == 168 or Ebeam == 172:
    #     shot = '50504'
    # else:
    #     shot = '50497'

    # shot = '49873'
    input_fname = 'input//II_a2&b2&a3&b3_' + shot + '.dat'
    print('\n>>INPUT FILE: ', input_fname)
    if input_fname != '':
        exp_voltages = np.loadtxt(input_fname)
        indexes = np.linspace(1, exp_voltages.shape[0]-1, NA2_points, dtype=int)

    if optimizeB2:
        optimizeA3B3 = True
        target = 'aim'
        # A2 plates voltage
        dUA2 = 2.
        UA2_range = np.arange(UA2min, UA2max + dUA2, dUA2)
        # UA2_range = np.linspace(UA2min, UA2max, NA2_points)  # [kV]
        eps_xy, eps_z = 1e-3, 1e-3
    else:
        target = 'aim'
        UA2_range = exp_voltages[indexes, 1]
        UB2_range = exp_voltages[indexes, 2]
        eps_xy, eps_z = 1e-3, 1.
    if not optimizeA3B3:
        target = 'aim'
        UA3_range = exp_voltages[indexes, 3]
        UB3_range = exp_voltages[indexes, 4]
        eps_xy, eps_z = 1e-3, 1.
    if optimizeB2:
        print('\n Primary beamline optimization')
    else:
        print('\n Calculating primary beamline')

    # UA2 loop
    for i in range(UA2_range.shape[0]):
        UA2 = UA2_range[i]
        if not optimizeB2:
            UB2 = UB2_range[i]
        if not optimizeA3B3:
            UA3, UB3 = UA3_range[i], UB3_range[i]
        print('\n\nE = {} keV; UA2 = {:.2f} kV\n'.format(Ebeam, UA2))
        # dict of starting voltages
        U_dict = {'A1': UA1, 'B1': UB1, 'A2': UA2, 'B2': UB2,
                  'A3': UA3, 'B3': UB3, 'A4': UA4, 'B4': UB4,
                  'an': Ebeam/(2*G)}
        # create new trajectory object
        tr = hb.Traj(q, m_ion, Ebeam, r0, geomTJ2.angles_dict['r0'][0],
                     geomTJ2.angles_dict['r0'][1], U_dict, dt)
        # optimize B2 voltage
        tr = hb.optimize_B2(tr, geomTJ2, UB2, dUB2, E, B, dt, stop_plane_n,
                            target, optimizeB2, eps_xy=eps_xy, eps_z=eps_z)
        UB2 = tr.U['B2']
        # check geometry intersection
        if True in tr.IntersectGeometry.values():
            print('NOT saved, primary intersected geometry')
            continue
        if True in tr.IntersectGeometrySec.values():
            print('NOT saved, secondary intersected geometry')
            continue
        if tr.IsAimXY and tr.IsAimZ:
            traj_list_B2.append(tr)
            print('\n Trajectory saved, UB2={:.2f} kV'.format(tr.U['B2']))
        else:
            print('NOT saved, sth wrong')
        # traj_list_B2.append(tr)

    t2 = time.time()
    if optimizeB2:
        print('\n B2 voltage optimized, t = {:.1f} s\n'.format(t2-t1))
    else:
        print('\n Trajectories to r_aim calculated, t = {:.1f} s\n'.format(t2-t1))

# %%
traj_list_passed = copy.deepcopy(traj_list_B2)

# %% Save traj list
# hb.save_traj_list(traj_list_passed, config, geomTJ2.r_dict[target])

# %% Additional plots
hbplot.plot_grid(traj_list_passed, geomTJ2, config, onlyE=True,
                 linestyle_A2='', marker_A2='')
# hbplot.plot_fan(traj_list_passed, geomTJ2, 132., UA2, config,
#                 plot_analyzer=False, plot_traj=True, plot_all=False)

hbplot.plot_scan(traj_list_passed, geomTJ2, Ebeam, config,
                 full_primary=False, plot_analyzer=True,
                 plot_det_line=True, subplots_vertical=True, scale=5)
# hbplot.plot_sec_angles(traj_list_passed, config, Ebeam='all')

# %% Optimize Secondary Beamline
t1 = time.time()
# define list of trajectories that hit slit
traj_list_a3b3 = []
if optimizeA3B3:
    print('\n Secondary beamline optimization')
    for tr in copy.deepcopy(traj_list_passed):
        tr, vltg_fail = hb.optimize_A3B3(tr, geomTJ2, UA3, UB3, dUA3, dUB3,
                                         E, B, dt, target='slit',
                                         UA3_max=40., UB3_max=40.,
                                         eps_xy=1e-3, eps_z=1e-3)
        # check geometry intersection and voltage failure
        if not (True in tr.IntersectGeometrySec.values()) and not vltg_fail:
            traj_list_a3b3.append(tr)
            print('\n Trajectory saved')
            # UA3 = tr.U['A3']
            # UB3 = tr.U['B3']
        else:
            print('\n NOT saved')
    t2 = time.time()
    print('\n A3 & B3 voltages optimized, t = {:.1f} s\n'.format(t2-t1))
else:
    print('\n Calculating secondary beamline')
    for tr in copy.deepcopy(traj_list_passed):
        print('\nEb = {}, UA2 = {:.2f}'.format(tr.Ebeam, tr.U['A2']))
        RV0 = np.array([tr.RV_sec[0]])
        tr.pass_sec(RV0, geomTJ2.r_dict['slit'], E, B, geomTJ2,
                    stop_plane_n=geomTJ2.plates_dict['an'].det_plane_n,
                    tmax=9e-5, eps_xy=eps_xy, eps_z=eps_z)
        traj_list_a3b3.append(tr)
    t2 = time.time()
    print('\n Secondary beamline calculated, t = {:.1f} s\n'.format(t2-t1))

# %% Pass to ANALYZER
if pass2AN:
    print('\n Passing to ANALYZER {}'.format(analyzer))
    # define list of trajectories that hit detector
    traj_list_an = []
    for tr in copy.deepcopy(traj_list_a3b3):
        print('\nEb = {}, UA2 = {:.2f}'.format(tr.Ebeam, tr.U['A2']))
        RV0 = np.array([tr.RV_sec[0]])
        # pass secondary trajectory to detector
        tr.pass_sec(RV0, geomTJ2.r_dict['det'], E, B, geomTJ2,
                    stop_plane_n=geomTJ2.plates_dict['an'].det_plane_n,
                    tmax=9e-5, eps_xy=eps_xy, eps_z=eps_z)
        traj_list_an.append(tr)

# %% Additional plots
# hbplot.plot_grid_a3b3(traj_list_a3b3, geomTJ2, config,
#                       linestyle_A2='--', linestyle_E='-',
#                       marker_E='p')
# hbplot.plot_traj(traj_list_a3b3, geomTJ2, 132., 0.0, config,
#                   full_primary=False, plot_analyzer=True,
#                   subplots_vertical=True, scale=3.5)
hbplot.plot_scan(traj_list_a3b3, geomTJ2, Ebeam, config,
                 full_primary=False, plot_analyzer=True,
                 plot_det_line=True, subplots_vertical=True, scale=4)

hbplot.plot_scan(traj_list_an, geomTJ2, Ebeam, config,
                 full_primary=False, plot_analyzer=True,
                 plot_det_line=True, subplots_vertical=True, scale=4)

# %% Pass trajectory to the Analyzer and find A4 voltage
#     print('\n Optimizing entrance angle to Analyzer with A4')
#     t1 = time.time()
#     traj_list_a4 = []
#     for tr in copy.deepcopy(traj_list_a3b3):
#         tr = hb.optimize_A4(tr, geomTJ2, UA4, dUA4,
#                             E, B, dt, eps_alpha=0.05)
#         if not tr.IntersectGeometrySec:
#             traj_list_a4.append(tr)
#             print('\n Trajectory saved')
#             UA4 = tr.U['A4']

#     t2 = time.time()
#     print("\n Calculation finished, t = {:.1f} s\n".format(t2-t1))

# %%
# hbplot.plot_traj(traj_list_a4, geomTJ2, 132., 0.0, config,
#                   full_primary=False, plot_analyzer=True)
# hbplot.plot_scan(traj_list_a4, geomTJ2, 132., config,
#                   full_primary=False, plot_analyzer=False,
#                   plot_det_line=False, subplots_vertical=True, scale=5)

# %% Save radref
if save_radref:
    Elist = [tr.Ebeam for tr in traj_list_passed]
    Elist = np.unique(Elist)
    for Ebeam in Elist:
        hb.save_radref(traj_list_passed, Ebeam, rho_interp)

# %% Save list of trajectories
# hb.save_traj_list(traj_list_passed, config, geomTJ2.r_dict[target])
