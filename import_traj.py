'''
Import lists with precalculated trajectories
'''
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import hibplib as hb
import hibpplotlib as hbplot
import copy

# %% import trajectory lists
config = '100_44_64'
alpha, beta = 73.3, -11.7
x, y, z = 270, -45, -17

traj_list = []

# grid with no shift
# # 18 dec 2019
# names = ['E92-100_UA2-5-2_alpha73_beta-12_x270y-45z-17.pkl',
#           'E104-112_UA2-6-2_alpha73_beta-12_x270y-45z-17.pkl',
#           'E116-124_UA2-7-2_alpha73_beta-12_x270y-45z-17.pkl',
#           'E128-136_UA2-8-2_alpha73_beta-12_x270y-45z-17.pkl',
#           'E140-148_UA2-9-2_alpha73_beta-12_x270y-45z-17.pkl',
#           'E150-150_UA2-9-2_alpha73_beta-12_x270y-45z-17.pkl']

# # mar 2020
# names = ['E92-104_UA2-5-3_alpha72.8_beta-11.7_x270y-45z-17.pkl',
#           'E108-120_UA2-6-3_alpha72.8_beta-11.7_x270y-45z-17.pkl',
#           'E124-136_UA2-8-3_alpha72.8_beta-11.7_x270y-45z-17.pkl',
#           'E140-152_UA2-9-3_alpha72.8_beta-11.7_x270y-45z-17.pkl',
#           'E156-168_UA2-9-3_alpha72.8_beta-11.7_x270y-45z-17.pkl',
#           'E172-176_UA2-9-3_alpha72.8_beta-11.7_x270y-45z-17.pkl']

# # mar 2020 5 cm up
# names = ['E92-104_UA2-5-3_alpha73_beta-12_x270y-40z-17.pkl',
#           'E108-120_UA2-6-3_alpha73_beta-12_x270y-40z-17.pkl',
#           'E124-136_UA2-8-3_alpha73_beta-12_x270y-40z-17.pkl',
#           'E140-152_UA2-9-3_alpha73_beta-12_x270y-40z-17.pkl',
#           'E156-168_UA2-9-3_alpha73_beta-12_x270y-40z-17.pkl',
#           'E172-176_UA2-9-3_alpha73_beta-12_x270y-40z-17.pkl']


# # apr 2019
names = [f'E100-108_UA2-6-3_alpha{alpha:.1f}_beta{beta:.1f}_x{x:.0f}y{y:.0f}z{z:.0f}.pkl',
         f'E112-120_UA2-7-2_alpha{alpha:.1f}_beta{beta:.1f}_x{x:.0f}y{y:.0f}z{z:.0f}.pkl',
         f'E124-132_UA2-8-2_alpha{alpha:.1f}_beta{beta:.1f}_x{x:.0f}y{y:.0f}z{z:.0f}.pkl',
         f'E136-144_UA2-8-2_alpha{alpha:.1f}_beta{beta:.1f}_x{x:.0f}y{y:.0f}z{z:.0f}.pkl',
         f'E148-150_UA2-8-2_alpha{alpha:.1f}_beta{beta:.1f}_x{x:.0f}y{y:.0f}z{z:.0f}.pkl']

for name in names:
    traj_list += hb.read_traj_list(name, dirname='output/'+config)

# %%
traj_list_passed = copy.deepcopy(traj_list)

# %% Save traj list
r_aim = geomTJ2.r_dict['aim']
hb.save_traj_list(traj_list_passed, config, r_aim)

# %% Additonal plots
hbplot.plot_grid(traj_list_passed, geomTJ2, config, onlyE=True)
