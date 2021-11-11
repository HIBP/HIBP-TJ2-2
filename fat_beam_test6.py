import numpy as np
import hibplib as hb
import hibpplotlib as hbplot
import copy
import matplotlib.pyplot as plt
import time

# %%
''' test FAT beam with focusing
'''
Ebeam = 132.
UA2 = 2.0

# n_slits = 5
# add slits to Geometry
# geomTJ2.add_slits(n_slits=n_slits, slit_dist=0.012, slit_w=4e-3,
#                   slit_l=0.1, slit_gamma=20.)
slits = range(5)  # [2]
r_slits = geomTJ2.plates_dict['an'].slits_edges
rs = geomTJ2.r_dict['slit']
# calculate normal to slit plane
slit_plane_n = geomTJ2.plates_dict['an'].slit_plane_n

# %%
traj_list_copy = copy.deepcopy(traj_list_a3b3)
# traj_list_copy = copy.deepcopy(traj_list_passed)

# %%
# set number of filaments in a beam
d_beam = 0.01  # beam diameter [m]
n_filaments_xy = 3  # number of filaments in xy plane (must be ODD)
skip_center_traj = True
n_gamma = 4  # number of chords in beam cross-section
foc_len = 50.  # distance from the first point of the trajectory to the focus
drad = np.pi/180.  # converts degrees to radians

fat_beam_list = []

t1 = time.time()
for tr in traj_list_copy:
    if tr.Ebeam == Ebeam and abs(tr.U['A2'] - UA2) < 0.1:
        print('\nEb = {}, UA2 = {}'.format(tr.Ebeam, tr.U['A2']))
    else:
        continue
    r0 = tr.RV0[0, :3]
    v0_abs = np.linalg.norm(tr.RV0[0, 3:])
    for i in range(n_filaments_xy):
        # skip center traj
        if abs(i - (n_filaments_xy-1)/2) < 1e-6 and skip_center_traj:
            continue
        # beam convergence angle
        alpha_conv = np.arctan((i - (n_filaments_xy-1)/2) *
                               (d_beam/(n_filaments_xy-1)) / foc_len)
        # set coords and velocity at the center of coord system
        x = 0.0
        y = -(i - (n_filaments_xy-1)/2) * (d_beam/(n_filaments_xy-1))
        z = 0.0
        r = np.array([x, y, z])
        print('XY filament number = {}'.format(i+1))
        v0 = v0_abs * np.array([-np.cos(alpha_conv),
                                np.sin(alpha_conv), 0.])
        # for gamma in np.arange(np.pi/n_gamma, np.pi, np.pi/n_gamma):
        for gamma in np.arange(0, np.pi, np.pi/n_gamma):
            gamma = gamma/drad
            print('\n gamma = ', gamma)
            # rotate and translate r0 to beam starting point
            r_rot = hb.rotate(r, axis=(1, 0, 0), deg=gamma)
            r_rot = hb.rotate(r_rot, axis=(0, 0, 1), deg=tr.alpha)
            r_rot = hb.rotate(r_rot, axis=(0, 1, 0), deg=tr.beta)
            r_rot += r0
            v_rot = hb.rotate(v0, axis=(1, 0, 0), deg=gamma)
            v_rot = hb.rotate(v_rot, axis=(0, 0, 1), deg=tr.alpha)
            v_rot = hb.rotate(v_rot, axis=(0, 1, 0), deg=tr.beta)

            tr_fat = copy.deepcopy(tr)
            tr_fat.RV0[0, :] = np.hstack([r_rot, v_rot])
            tr_fat.pass_fan(geomTJ2.r_dict['aim'], E, B, geomTJ2,
                            eps_xy=1e-3, eps_z=1,
                            stop_plane_n=np.array([1., 0., 0.]),
                            no_intersect=True, no_out_of_bounds=True)
            tr_fat.pass_to_target(geomTJ2.r_dict['aim'], E, B, geomTJ2,
                                  eps_xy=1e-3, eps_z=1,
                                  stop_plane_n=np.array([1., 0., 0.]),
                                  no_intersect=True, no_out_of_bounds=True)
            # tr_fat = hb.pass_to_slits(tr_fat, dt, E, B, geomTJ2,
            #                           timestep_divider=10)
            tr_fat = hb.calc_zones(tr_fat, dt, E, B, geomTJ2, slits=slits,
                                   timestep_divider=6,
                                   eps_xy=1e-4, eps_z=1, dt_min=1e-13,
                                   no_intersect=True, no_out_of_bounds=True)
            fat_beam_list.append(tr_fat)
            if abs(y) < 1e-6:
                break
t2 = time.time()
print('t = {:.1f} s\n'.format(t2-t1))

# %% save zones
# dirname = 'output/' + 'B{}_I{}'.format(int(Btor), int(Ipl))
# for i_slit in range(n_slits):
#     zone = np.empty([0, 3])
#     for tr in fat_beam_list:
#         zone = np.vstack([zone, tr.ion_zones[i_slit]])
#     fname = dirname + '/' + 'E{}'.format(int(Ebeam)) + \
#         '_UA2{}'.format(int(UA2)) + '_slit{}.txt'.format(i_slit)
#     np.savetxt(fname, zone, fmt='%.4e')

# %% plot fat beam
hbplot.plot_fat_beam(fat_beam_list, geomTJ2, config, slits=range(5))
hbplot.plot_fat_beam(fat_beam_list, geomTJ2, config, slits=slits)

# %% plot SVs
hbplot.plot_svs(fat_beam_list, geomTJ2, config, slits=range(5),
                plot_prim=True, plot_sec=False, plot_zones=False,
                plot_cut=False, alpha_xy=10, alpha_zy=20)
hbplot.plot_svs(fat_beam_list, geomTJ2, config, slits=slits,
                plot_prim=True, plot_sec=False, plot_zones=True,
                plot_cut=False, alpha_xy=10, alpha_zy=20)
