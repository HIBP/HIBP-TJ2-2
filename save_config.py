'''
import contours for magnetic configuration and save them to file
'''
import numpy as np
import matplotlib.pyplot as plt

config = '100_44_64'

fname = 'tj2lib//' + config + '_rz_Brho.txt'
data = np.loadtxt(fname)

r = np.unique(data[:, 0])
z = np.unique(data[:, 1])
R, Z = np.meshgrid(r, z)

fig, ax = plt.subplots()
cs = ax.contour(R, Z, data[:, 5].reshape(R.shape).T,
                levels=np.linspace(0.05, 0.9999, 15))
ax.axis('equal')

# %%get contours coordinates and contour levels
conts = cs.allsegs
levels = cs.levels
# save to file
output_file = open('configs//' + config + '.txt', 'w')
for i in range(len(conts)):
    output_file.write('{:.3f} nan\n'.format(levels[i]))
    np.savetxt(output_file, conts[i][0])
output_file.close()
