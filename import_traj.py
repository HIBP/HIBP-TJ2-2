import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import hibplib as hb
import hibpplotlib as hbplot
import copy

# %% import trajectory lists
config = '100_44_64'
traj_list = []

# grid with alpha=30
names = ['E92-152_UA2-9-2_alpha73_beta-12_x270y-45z-17.pkl',
         'E150-150_UA2-8-2_alpha73_beta-12_x270y-45z-17.pkl']

# grid with alpha=30 5 cm up
names = ['E92-148_UA2-9-2_alpha73_beta-12_x270y-40z-17.pkl',
         'E150-150_UA2-9-2_alpha73_beta-12_x270y-40z-17.pkl']

# grid with alpha=30 10 cm up
names = ['E92-104_UA2-6-2_alpha73_beta-12_x270y-35z-17.pkl',
         'E108-120_UA2-7-2_alpha73_beta-12_x270y-35z-17.pkl',
         'E124-136_UA2-8-2_alpha73_beta-12_x270y-35z-17.pkl',
         'E140-148_UA2-9-2_alpha73_beta-12_x270y-35z-17.pkl',
         'E150-150_UA2-9-2_alpha73_beta-12_x270y-35z-17.pkl']

# grid with alpha=30 8 cm up and 2 cm right
names = ['E92-104_UA2-5-2_alpha73_beta-12_x272y-37z-17.pkl',
         'E108-120_UA2-6-2_alpha73_beta-12_x272y-37z-17.pkl',
         'E124-136_UA2-8-2_alpha73_beta-12_x272y-37z-17.pkl',
         'E140-148_UA2-9-2_alpha73_beta-12_x272y-37z-17.pkl',
         'E150-150_UA2-9-2_alpha73_beta-12_x272y-37z-17.pkl']

for name in names:
    traj_list += hb.read_traj_list(name, dirname='output/'+config)

# %%
traj_list_passed = copy.deepcopy(traj_list)

# %% Save traj list
r_aim = geomTJ2.r_dict['aim1']
hb.save_traj_list(traj_list_passed, config, r_aim)

# %% Additonal plots
hbplot.plot_grid(traj_list_passed, geomTJ2, config, onlyE=True)
