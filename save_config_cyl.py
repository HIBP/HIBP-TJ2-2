'''
import contours for magnetic configuration and save them to file
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

config = '100_44_64'

fname = 'tj2lib//' + config + '_rz_Brho_cyl.txt'
data = np.loadtxt(fname)

r, z, rho = data[:, 0], data[:, 1], data[:, 5]

# cteate grid
delta = 1e-3
ri = np.arange(1.1, 1.7, delta)
zi = np.arange(-0.35, 0.25, delta)
Ri, Zi = np.meshgrid(ri, zi)

# Perform linear interpolation of the data
# on a grid defined by (r, z)
triang = tri.Triangulation(r, z)
interpolator = tri.LinearTriInterpolator(triang, rho)

rhoi = interpolator(Ri, Zi)

# ax1.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
# cntr1 = ax1.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")

# fig.colorbar(cntr1, ax=ax1)

fig, ax = plt.subplots()
cs = ax.contour(ri, zi, rhoi, levels=np.linspace(0.1, 1, 20))
ax.axis('equal')

# %% get contours coordinates and contour levels
conts = cs.allsegs
levels = cs.levels

# %% save to file
# output_file = open('configs//' + config + '.txt', 'w')
# for i in range(len(conts)):
#     output_file.write('{:.3f} nan\n'.format(levels[i]))
#     np.savetxt(output_file, conts[i][0])
# output_file.close()
