import numpy as np
import hibplib as hb
import hibpplotlib as hbplot
import copy

# %%
''' pass trajectories to different slits
'''
Ebeam = 132.
UA2 = -1.9

n_slits = 7
# add slits to Geometry
# geomT15.add_slits(n_slits=n_slits, slit_dist=0.01, slit_w=5e-3,
#                   slit_l=0.1)  # -20.)
r_slits = geomTJ2.slits_edges
rs = geomTJ2.r_dict['slit']
# calculate normal to slit plane
slit_plane_n = geomTJ2.slit_plane_n

# %%
# traj_list_copy = copy.deepcopy(traj_list_a3b3)
traj_list_copy = copy.deepcopy(traj_list_passed)

# %%
print('\n*** Passing fan to {} slits'.format(n_slits))
for tr in traj_list_copy:
    if abs(tr.Ebeam - Ebeam) < 0.1 and abs(tr.U['A2'] - UA2) < 0.1:
        print('\nEb = {}, UA2 = {}'.format(tr.Ebeam, tr.U['A2']))
    else:
        continue

    tr = hb.pass_to_slits(tr, dt, E, B, geomTJ2, timestep_divider=5)
    break

# %% plot trajectories
hbplot.plot_traj_toslits(tr, geomTJ2, config, plot_fan=True)
