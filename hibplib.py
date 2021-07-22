import numpy as np
import os
import errno
import pickle as pc
import hibpplotlib as hbplot
import copy
from matplotlib import path
from scipy.interpolate import RegularGridInterpolator
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from itertools import cycle
import numba


# %% define class for trajectories

class Traj():

    def __init__(self, q, m, Ebeam, r0, alpha, beta, U, dt=1e-7):
        ''' class for trajectories
        q - particle charge [Co]
        m - particle mass [kg]
        Ebeam - beam energy [keV]
        r0 - initial position [m]
        alpha - injection angle in XY plane [rad]
        beta - injection angle in XZ plane [rad]
        U - dict of voltages in [kV] keys=[A1 B1 A2 B2 A3 B3 an]
        dt - timestep for RK algorithm [s]
        '''
        self.q = q
        self.m = m
        self.Ebeam = Ebeam
        # particle velocity:
        Vabs = np.sqrt(2 * Ebeam * 1.602176634E-16 / m)
        V0 = calc_vector(Vabs, alpha, beta, direction=(-1, -1, 1))
        self.alpha = alpha
        self.beta = beta
        self.U = U
        self.RV0 = np.array([np.hstack((r0, V0))])  # initial condition

        self.RV_prim = self.RV0  # array with r,V for the whole trajectory
        self.tag_prim = [1]

        self.RV_sec = np.array([[]])
        self.tag_sec = [2]

        # list to contain RV of the whole fan:
        self.Fan = []
        # time step for primary orbit:
        self.dt1 = dt
        # time step for secondary orbit:
        self.dt2 = dt
        self.IsAimXY = False
        self.IsAimZ = False
        self.IntersectGeometry = {'A2': False, 'B2': False, 'chamb': False}
        self.IntersectGeometrySec = {'A3': False, 'B3': False, 'A4': False,
                                     'chamb': False}
        self.B_out_of_bounds = False
        # multislit:
        self.fan_to_slits = []
        self.RV_sec_toslits = []
        self.ion_zones = []

    def pass_prim(self, E_interp, B_interp, geom, tmax=1e-5):
        ''' passing primary trajectory from initial point self.RV0
            geom - Geometry object
        '''
        print('\n Passing primary trajectory')
        # reset intersection flags
        for key in self.IntersectGeometry.keys():
            self.IntersectGeometry[key] = False
        t = 0.
        dt = self.dt1
        RV_old = self.RV0  # initial position
        RV = self.RV0  # array to collect all r, V
        k = self.q / self.m
        tag_column = [10]

        while t <= tmax:
            r = RV_old[0, :3]
            # Electric field:
            E_local = return_E(r, E_interp, self.U, geom)
            # Magnetic field:
            try:
                B_local = return_B(r, B_interp)
            except ValueError:
                print('Btor Out of bounds for primaries, r = ', r)
                print(' t = ', t)
                break
            # runge-kutta step:
            RV_new = runge_kutt(k, RV_old, dt, E_local, B_local)
            RV = np.vstack((RV, RV_new))

            tag_column = np.hstack((tag_column, 10))

            if geom.check_chamb_intersect('prim', RV_old[0, 0:3], RV_new[0, 0:3]):
                print('Primary intersected chamber')
                self.IntersectGeometry['chamb'] = True
                break

            plts_flag, plts_name = geom.check_plates_intersect(RV_old[0, 0:3],
                                                               RV_new[0, 0:3])
            if plts_flag:
                print('Primary intersected ' + plts_name + ' plates')
                self.IntersectGeometry[plts_name] = True
                break

            RV_old = RV_new
            t = t + dt
            # print('t = ', t)

        self.RV_prim = RV
        self.tag_prim = tag_column

    def pass_sec(self, RV0, r_aim, E_interp, B_interp, geom,
                 stop_plane_n=np.array([1, 0, 0]), tmax=5e-5,
                 eps_xy=1e-3, eps_z=1e-3):
        ''' passing secondary trajectory from initial point RV0 to point r_aim
            with accuracy eps
            geom - Geometry object
        '''
        # print('Passing secondary trajectory')
        self.IsAimXY = False
        self.IsAimZ = False
        self.B_out_of_bounds = False
        # reset intersection flags
        for key in self.IntersectGeometrySec.keys():
            self.IntersectGeometrySec[key] = False
        t = 0.
        dt = self.dt2
        RV_old = RV0  # initial position
        RV = RV0  # array to collect all [r,V]
        k = 2*self.q / self.m
        tag_column = [20]

        while t <= tmax:  # to witness curls initiate tmax as 1 sec
            r = RV_old[0, :3]
            # Electric field:
            E_local = return_E(r, E_interp, self.U, geom)
            # Magnetic field:
            try:
                B_local = return_B(r, B_interp)
            except ValueError:
                print('Btor Out of bounds for secondaries, r = ',
                      np.round(r, 3))
                print(' t = ', t)
                self.B_out_of_bounds = True
                break
            # runge-kutta step:
            RV_new = runge_kutt(k, RV_old, dt, E_local, B_local)

            if geom.check_chamb_intersect('sec', RV_old[0, 0:3], RV_new[0, 0:3]):
                # print('Secondary intersected chamber exit')
                self.IntersectGeometrySec['chamb'] = True

            plts_flag, plts_name = geom.check_plates_intersect(RV_old[0, 0:3],
                                                               RV_new[0, 0:3])
            if plts_flag:
                print('Secondary intersected ' + plts_name + ' plates')
                self.IntersectGeometrySec[plts_name] = True

            # find last point of the secondary trajectory
            if (RV_new[0, 0] > 1.55) and (RV_new[0, 1] < 0.0):
                # intersection with the stop plane:
                planeNormal = stop_plane_n
                planePoint = r_aim
                rayDirection = RV_new[0, :3] - RV_old[0, :3]
                rayPoint = RV_new[0, :3]
                r_intersect = line_plane_intersect(planeNormal, planePoint,
                                                   rayDirection, rayPoint)
                # check if r_intersect is between RV_old and RV_new:
                if is_between(RV_old[0, :3], RV_new[0, :3], r_intersect):
                    RV_new[0, :3] = r_intersect
                    RV = np.vstack((RV, RV_new))
                    # check XY plane:
                    if (np.linalg.norm(RV_new[0, :2] - r_aim[:2]) <= eps_xy):
                        # print('aim XY!')
                        self.IsAimXY = True
                    # check XZ plane:
                    if (np.linalg.norm(RV_new[0, [0, 2]] - r_aim[[0, 2]]) <=
                            eps_z):
                        # print('aim Z!')
                        self.IsAimZ = True
                    break

            # continue trajectory calculation:
            RV_old = RV_new
            t = t + dt
            RV = np.vstack((RV, RV_new))
            tag_column = np.hstack((tag_column, 20))
            # print('t secondary = ', t)

        self.RV_sec = RV
        self.tag_sec = tag_column

    def pass_fan(self, r_aim, E_interp, B_interp, geom,
                 stop_plane_n=np.array([1, 0, 0]), eps_xy=1e-3, eps_z=1e-3,
                 no_intersect=False, no_out_of_bounds=False):
        ''' passing fan from initial point self.RV0
        geom - geometry object
        '''
        print('\n Passing fan of trajectories')
        self.pass_prim(E_interp, B_interp, geom)
        # create a list fro secondary trajectories:
        list_sec = []
        # check intersection of primary trajectory:
        if True in self.IntersectGeometry.values():
            print('Fan list is empty')
            self.Fan = list_sec
            return 0

        # check eliptical radius of particle:
        # 1.5 m - major radius of a torus, elon - size along Y
        mask = np.sqrt((self.RV_prim[:, 0] - geom.R)**2 +
                       (self.RV_prim[:, 1] / geom.elon)**2) <= geom.r_plasma
        self.tag_prim[mask] = 11

        # list of initial points of secondary trajectories:
        RV0_sec = self.RV_prim[(self.tag_prim == 11)]

        for RV02 in RV0_sec:
            RV02 = np.array([RV02])
            self.pass_sec(RV02, r_aim, E_interp, B_interp, geom,
                          stop_plane_n=stop_plane_n,
                          eps_xy=eps_xy, eps_z=eps_z)
            if (no_intersect and True in self.IntersectGeometrySec.values()) or \
               (no_out_of_bounds and self.B_out_of_bounds):
                continue
            list_sec.append(self.RV_sec)

        self.Fan = list_sec

    def add_slits(self, n_slits):
        ''' empty list for secondary trajectories, which go to different slits
        '''
        if len(self.RV_sec_toslits) == n_slits:
            pass
        else:
            self.RV_sec_toslits = [None]*n_slits
            self.ion_zones = [None]*n_slits

    def plot_prim(self, ax, axes='XY', color='k', full_primary=False):
        ''' plot primary trajectory
        '''
        axes_dict = {'XY': (0, 1), 'XZ': (0, 2), 'ZY': (2, 1)}
        index_X, index_Y = axes_dict[axes]
        index = -1
        if not full_primary:
            # find where secondary trajectory starts:
            for i in range(self.RV_prim.shape[0]):
                if np.linalg.norm(self.RV_prim[i, :3]
                                  - self.RV_sec[0, :3]) < 1e-4:
                    index = i+1
        ax.plot(self.RV_prim[:index, index_X],
                self.RV_prim[:index, index_Y],
                color=color, linewidth=2)

    def plot_sec(self, ax, axes='XY', color='r'):
        ''' plot secondary trajectory
        '''
        axes_dict = {'XY': (0, 1), 'XZ': (0, 2), 'ZY': (2, 1)}
        index_X, index_Y = axes_dict[axes]
        ax.plot(self.RV_sec[:, index_X], self.RV_sec[:, index_Y],
                color=color, linewidth=2)

    def plot_fan(self, ax, axes='XY', color='r'):
        ''' plot fan of secondary trajectories
        '''
        axes_dict = {'XY': (0, 1), 'XZ': (0, 2), 'ZY': (2, 1)}
        index_X, index_Y = axes_dict[axes]
        for i in self.Fan:
            ax.plot(i[:, index_X], i[:, index_Y], color=color)


# %% define class for geometry
class Geometry():
    '''
    object containing geometry points
    '''

    def __init__(self):
        # lists with coordinates of 4 points,
        # determining chamber entrance and exit:
        self.chamb_ent = []
        self.chamb_ext = []
        # dictionary for arrays of plates coordinates:
        self.plates_edges = dict()
        # dictionary for positions of all plates:
        self.r_dict = dict()
        # vacuum vessel contour:
        self.camera = np.array([])
        # separatrix contour:
        self.sep = np.array([])
        # inner and outer first wall contours:
        self.in_fw = np.array([])
        self.out_fw = np.array([])
        # array for slits coordinates:
        self.slits_edges = np.array([])
        # array for slit normal:
        self.slit_plane_n = np.array([])
        # array for slit polygon:
        self.slits_spot = np.array([])
        # arrays for primary and secondary beamline angles:
        self.angles = dict()
        # array for Analyzer parameters
        self.an_params = np.array([])
        # plasma geometry
        self.R = 0
        self.r_plasma = 0
        self.elon = 0

    def check_chamb_intersect(self, beamline, point1, point2):
        ''' check intersection between segment 1->2 and chamber
        '''
        intersect_flag = False
        # do not check intersection when particle is far from chamber
        if point1[0] > 1.95 and point2[1] > 0.7:
            return intersect_flag
        if beamline == 'prim':
            # check intersection with chamber entrance and chamber at HFS
            # if len(self.chamb_ent) == 0: return False
            for coords_list in [self.chamb_ent, self.chamb]:
                flags_list = [segm_intersect(point1[0:2], point2[0:2],
                                             coords_list[i], coords_list[i+1])
                              for i in np.arange(0, len(coords_list), 2)]
                intersect_flag = intersect_flag or (True in flags_list)
        elif beamline == 'sec':
            # check intersection with chamber exit
            # if len(self.chamb_ext) == 0: return False
            for i in np.arange(0, len(self.chamb_ext), 2):
                intersect_flag = intersect_flag or \
                    segm_intersect(point1[0:2], point2[0:2],
                                   self.chamb_ext[i], self.chamb_ext[i+1])
        return intersect_flag

    def check_plates_intersect(self, point1, point2):
        # do not check intersection when particle is outside beamlines
        if point2[0] < self.r_dict['aim1'][0]-0.05 and \
           point1[1] < self.r_dict['port'][1]:
            return False, 'none'
        segment_coords = np.array([point1, point2])
        for key in self.plates_edges.keys():
            # check if a point in inside the beamline
            if (key in ['A1', 'B1', 'A2', 'B2'] and
                point1[1] > self.r_dict['port'][1]) or \
                (key in ['A3', 'B3', 'A4', 'B4'] and
                 point2[0] > self.r_dict['aim1'][0]-0.05):
                # check intersection
                if segm_poly_intersect(self.plates_edges[key][0],
                                       segment_coords) or \
                    segm_poly_intersect(self.plates_edges[key][1],
                                        segment_coords):
                    return True, key
            else:
                continue
        return False, 'none'

    def add_coords(self, name, ref_point, dist, angles):
        ''' add new element 'name' to r_dict
        '''
        # unpack ref_point
        if type(ref_point) == str:
            r0 = self.r_dict[ref_point]
        else:
            r0 = ref_point
        # unpack angles
        alpha, beta = angles[0:2]
        # coordinates of the center of the object
        r = r0 + calc_vector(dist, alpha, beta)
        self.r_dict[name] = r

    def add_slits(self, n_slits, slit_dist, slit_w, slit_l):
        ''' add slits to Geometry
        n_slits - number of slits
        slit_dist - distance between centers of the slits [m]
        slit_w - slit width (along Y) [m]
        slit_l - slit length (along Z)
        slit_gamma - angle of ratation around X [deg]
        '''
        # angles of the slits plane normal
        slit_angles = copy.deepcopy(self.angles['an'])
        # coords of center of the central slit
        rs = self.r_dict['slit']

        r_slits, slit_plane_n, slits_spot = define_slits(rs, slit_angles,
                                                         n_slits, slit_dist,
                                                         slit_w, slit_l)
        self.slits_edges = r_slits
        self.slit_plane_n = slit_plane_n
        self.slits_spot = slits_spot

    def add_detector(self, n_det, det_dist, det_w, det_l):
        ''' add detector to geometry
        '''
        n_det = int(n_det)
        if self.an_params.shape[0] == 0:
            print('Analyzer not defined!')
            return
        # analyzer parameters
        XD, YD1, YD2 = self.an_params[5:]
        theta_an = self.an_params[4]
        # set detector angles
        det_angles = np.array([0, 0, 0])
        det_angles[0] = det_angles[0] + 180. - theta_an
        r_det, det_plane_n, det_spot = define_slits(np.array([np.sqrt(XD**2 + (YD1 - YD2)**2), 0, 0]),
                                                    det_angles,
                                                    n_det, det_dist,
                                                    det_w, det_l)
        # angles of the beamline
        beamline_angles = copy.deepcopy(self.angles['an'])
        # angles of the analyzer
        angles = copy.deepcopy(self.angles['an'])
        angles[0] = angles[0] - theta_an
        # rotate and shift to position
        for i_slit in range(n_det):
            for j in range(5):
                r_det[i_slit, j, :] = rotate3(r_det[i_slit, j, :],
                                              angles, beamline_angles)
                r_det[i_slit, j, :] += self.r_dict['slit']
            if n_det//2 - i_slit == 0:
                # add coords of the center of the central detector
                self.add_coords('det', r_det[i_slit, 0, :], 0, [0, 0])

        # calculate normal to slit plane:
        det_plane_n = np.cross(r_det[0, 0, :] - r_det[0, 1, :],
                               r_det[0, 0, :] - r_det[0, 2, :])
        det_plane_n = det_plane_n/np.linalg.norm(det_plane_n)

        # create polygon, which contains all slits (slits spot):
        r0 = self.r_dict['det']
        det_spot = 1.5*np.vstack([r_det[0, [1, 4], :] - r0,
                                  r_det[-1, [3, 2], :] - r0]) + r0
        self.det_edges = r_det
        self.det_plane_n = det_plane_n
        self.det_spot = det_spot

    def plot_geom(self, ax, axes='XY', plot_sep=True, plot_aim=True):
        '''
        plot camera, plates, aim dot and central slit dot
        '''
        # plot toroidal and poloidal field coils, camera and
        # separatrix in XY plane
        if axes == 'XY':
            # plot camera
            ax.plot(self.camera[:, 0] + self.R, self.camera[:, 1],
                    color='tab:blue')

            if plot_sep:
                ax.plot(self.sep[:, 0], self.sep[:, 1],
                        markersize=2, color='tab:gray')  # 'tab:orange')

        index_X, index_Y = get_index(axes)
        # plot plates
        for name in self.plates_edges.keys():
            if name == 'an':
                continue  # do not plot Analyzer
            ax.fill(self.plates_edges[name][0][:, index_X],
                    self.plates_edges[name][0][:, index_Y], fill=False,
                    hatch='\\', linewidth=2)
            ax.fill(self.plates_edges[name][1][:, index_X],
                    self.plates_edges[name][1][:, index_Y], fill=False,
                    hatch='/', linewidth=2)

        if plot_aim:
            # plot aim dot
            ax.plot(self.r_dict['aim'][index_X], self.r_dict['aim'][index_Y],
                    '*', color='b')
            # plot the center of the central slit
            ax.plot(self.r_dict['slit'][index_X], self.r_dict['slit'][index_Y],
                    '*', color='g')

    def plot_analyzer(self, ax, axes='XY', n_slit='all', color='g'):
        index_X, index_Y = get_index(axes)
        # plot plates
        for name in self.plates_edges.keys():
            if name == 'an':
                ax.fill(self.plates_edges[name][0][:, index_X],
                        self.plates_edges[name][0][:, index_Y], fill=False,
                        hatch='\\', linewidth=2)
                ax.fill(self.plates_edges[name][1][:, index_X],
                        self.plates_edges[name][1][:, index_Y], fill=False,
                        hatch='/', linewidth=2)
        # plot slits
        plot_slits(self.slits_edges, self.slits_spot, ax, axes=axes,
                   n_slit=n_slit, color=color)
        # plot detector
        plot_slits(self.det_edges, self.det_spot, ax, axes=axes,
                   n_slit=n_slit, color=color)


# %%
def plot_slits(r_slits, spot, ax, axes='XY', n_slit='all', color='g'):
    ''' plot slits contours
    '''
    index_X, index_Y = get_index(axes)

    if n_slit == 'all':
        slits = range(r_slits.shape[0])
    else:
        slits = [n_slit]

    # set color cycler
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = colors[:len(slits)]
    colors = cycle(colors)

    for i in slits:
        c = next(colors)
        # plot center
        ax.plot(r_slits[i, 0, index_X], r_slits[i, 0, index_Y],
                '*', color=c)
        # plot edge
        ax.fill(r_slits[i, 1:, index_X], r_slits[i, 1:, index_Y],
                fill=False)
    # plot slits spot
    ax.fill(spot[:, index_X], spot[:, index_Y], fill=False)


# %%
def define_slits(r0, slit_angles, n_slits, slit_dist, slit_w, slit_l):
    n_slits = int(n_slits)
    # calculate slits coordinates:
    r_slits = np.zeros([n_slits, 5, 3])
    for i_slit in range(n_slits):
        # calculate coords of slit center:
        y0 = (n_slits//2 - i_slit)*slit_dist
        r_slits[i_slit, 0, :] = [0., y0, 0.]
        # calculate slit edges:
        r_slits[i_slit, 1, :] = [0., y0 + slit_w/2, slit_l/2]
        r_slits[i_slit, 2, :] = [0., y0 - slit_w/2, slit_l/2]
        r_slits[i_slit, 3, :] = [0., y0 - slit_w/2, -slit_l/2]
        r_slits[i_slit, 4, :] = [0., y0 + slit_w/2, -slit_l/2]
        # rotate and shift to slit position:
        for j in range(5):
            r_slits[i_slit, j, :] = rotate3(r_slits[i_slit, j, :],
                                            slit_angles, slit_angles)
            r_slits[i_slit, j, :] += r0

    # calculate normal to slit plane:
    slit_plane_n = np.cross(r_slits[0, 0, :] - r_slits[0, 1, :],
                            r_slits[0, 0, :] - r_slits[0, 2, :])
    slit_plane_n = slit_plane_n/np.linalg.norm(slit_plane_n)

    # create polygon, which contains all slits (slits spot):
    slits_spot = 1.5*np.vstack([r_slits[0, [1, 4], :] - r0,
                                r_slits[-1, [3, 2], :] - r0]) + r0

    return r_slits, slit_plane_n, slits_spot


# %%
@numba.jit()
def calc_vector(length, alpha, beta, direction=(1, 1, -1)):
    ''' calculate vector based on its length and angles
    '''
    drad = np.pi/180.  # converts degrees to radians
    x = direction[0] * length * np.cos(alpha*drad) * np.cos(beta*drad)
    y = direction[1] * length * np.sin(alpha*drad)
    z = direction[2] * length * np.cos(alpha*drad) * np.sin(beta*drad)
    return np.array([x, y, z])


# %%
@numba.jit()
def calc_angles(vector):
    ''' calculate alpha and beta angles based on vector coords
    '''
    drad = np.pi/180.  # converts degrees to radians
    x, y, z = vector / np.linalg.norm(vector)
    alpha = np.arcsin(y)  # rad
    if abs(y) > 1e-9:
        beta = np.arcsin(-np.tan(alpha) * z / y)  # rad
    elif abs(z) < 1e-9:
        beta = 0.
    elif abs(x) > 1e-9:
        beta = np.arctan(-z / x)  # rad
    else:
        beta = -np.sign(z) * np.pi/2
    return alpha/drad, beta/drad  # degrees


# %% get axes index
def get_index(axes):
    axes_dict = {'XY': (0, 1), 'XZ': (0, 2), 'ZY': (2, 1)}
    return axes_dict[axes]


# %% Runge-Kutta
# define equations of movement:

@numba.jit()
def f(k, E, V, B):
    return k*(E + np.cross(V, B))


@numba.jit()
def g(V):
    return V


@numba.jit()
def runge_kutt(k, RV, dt, E, B):
    '''
    Calculate one step using Runge-Kutta algorithm
    :param k: particle charge [Co] / particle mass [kg]
    :param RV: 7 dimensial vector
           array[[x,y,z,vx,vy,vz,Flag]]
           Flag = 10 primary out of plasma
           Flag = 11 primary in plasma
           Flag = 20 secondary
    :param dt: time discretisation step
    :param E: values of electric field in current point
              np.array([Ex, Ey, Ez])
    :param B: values of magnetic field in current point
              np.array([Bx, By, Bz])
    :return: new RV

     V' = k(E + [VxB]) == K(E + np.cross(V,B)) == f
     r' = V == g

    V[n+1] = V[n] + (h/6)(m1 + 2m2 + 2m3 + m4)
    r[n+1] = r[n] + (h/6)(k1 + 2k2 + 2k3 + k4)
    m[1] = f(t[n], V[n], r[n])
    k[1] = g(t[n], V[n], r[n])
    m[2] = f(t[n] + (h/2), V[n] + (h/2)m[1], r[n] + (h/2)k[1])
    k[2] = g(t[n] + (h/2), V[n] + (h/2)m[1], r[n] + (h/2)k[1])
    m[3] = f(t[n] + (h/2), V[n] + (h/2)m[2], r[n] + (h/2)k[2])
    k[3] = g(t[n] + (h/2), V[n] + (h/2)m[2], r[n] + (h/2)k[2])
    m[4] = f(t[n] + h, V[n] + h*m[3], r[n] + h*k[3])
    k[4] = g(t[n] + h, V[n] + h*m[3], r[n] + h*k[3])

    E - np.array([Ex, Ey, Ez])
    B - np.array([Bx, By, Bz])
    '''
    r = RV[0, :3]
    V = RV[0, 3:]

    ''' m1,k1 '''
    m1 = f(k, E, V, B)
    k1 = g(V)
    ''' m2,k2 '''
    fV2 = V + (dt / 2.) * m1
    gV2 = V + (dt / 2.) * m1
    m2 = f(k, E, fV2, B)
    k2 = g(gV2)
    ''' m3,k3 '''
    fV3 = V + (dt / 2.) * m2
    gV3 = V + (dt / 2.) * m2
    m3 = f(k, E, fV3, B)
    k3 = g(gV3)
    ''' m4,k4 '''
    fV4 = V + dt * m3
    gV4 = V + dt * m3
    m4 = f(k, E, fV4, B)
    k4 = g(gV4)
    ''' all together! '''
    V = V + (dt / 6.) * (m1 + (2. * m2) + (2. * m3) + m4)
    r = r + (dt / 6.) * (k1 + (2. * k2) + (2. * k3) + k4)

    RV = np.hstack((r, V))

    return RV


# %%
def optimize_B2(tr, geom, UB2, dUB2, E, B, dt, stop_plane_n, target='aim',
                optimize=True, eps_xy=1e-3, eps_z=1e-3):
    ''' get voltages on B2 plates and choose secondary trajectory
    which goes into r_aim
    '''
    print('Target: ' + target)
    r_aim = geom.r_dict[target]
    k = tr.q/tr.m
    attempts_high = 0
    attempts_low = 0
    attempts_opt = 0
    while True:
        tr.U['B2'], tr.dt1, tr.dt2 = UB2, dt, dt
        # pass fan of trajectories
        tr.pass_fan(r_aim, E, B, geom, stop_plane_n=stop_plane_n,
                    eps_xy=eps_xy, eps_z=eps_z,
                    no_intersect=True, no_out_of_bounds=True)
        if True in tr.IntersectGeometry.values():
            break
        if len(tr.Fan) == 0:
            print('NO secondary trajectories')
            break
        # reset flags in order to let the algorithm work properly
        tr.IsAimXY = False
        tr.IsAimZ = False
        # reset intersection flags
        for key in tr.IntersectGeometrySec.keys():
            tr.IntersectGeometrySec[key] = False

        # find which secondaries are higher/lower than r_aim
        # sign = -1 means higher, 1 means lower
        signs = np.array([np.sign(np.cross(RV[-1, :3], r_aim)[-1])
                          for RV in tr.Fan])
        are_higher = np.argwhere(signs == -1)
        are_lower = np.argwhere(signs == 1)
        twisted_fan = False

        if are_higher.shape[0] == 0:
            print('all secondaries are lower than aim!')
            attempts_high += 1
            n = int(are_lower[are_lower.shape[0]//2])
            if attempts_high > 5:
                attempts_high = 0
                print('Aim is too HIGH along Y!')
                # return tr
                break
        elif are_lower.shape[0] == 0:
            print('all secondaries are higher than aim!')
            attempts_low += 1
            n = int(are_higher[are_higher.shape[0]//2])
            if attempts_low > 5:
                attempts_low = 0
                print('Aim is too LOW along Y!')
                # return tr
                break
        else:
            attempts_high = 0
            attempts_low = 0
            if are_higher[-1] > are_lower[0]:
                print('Fan is twisted!')
                twisted_fan = True
                n = int(are_lower[-1])
            else:
                n = int(are_higher[-1])  # find the last one which is higher
        RV_old = np.array([tr.Fan[n][0]])

        # find secondary, which goes directly into r_aim
        tr.dt1 = tr.dt1/2.
        while True:
            # make a small step along primary trajectory
            r = RV_old[0, :3]
            try:
                B_local = return_B(r, B)
            except ValueError:
                print('Btor out of bounds while optimizing B2')
                break
            E_local = np.array([0., 0., 0.])
            RV_new = runge_kutt(k, RV_old, tr.dt1, E_local, B_local)
            # pass new secondary trajectory
            tr.pass_sec(RV_new, r_aim, E, B, geom,
                        stop_plane_n=stop_plane_n, eps_xy=eps_xy, eps_z=eps_z)
            # check XY flag
            if tr.IsAimXY:
                # insert RV_new into primary traj
                index = np.nanargmin(np.linalg.norm(tr.RV_prim[:, :3] -
                                                    RV_new[0, :3], axis=1))
                tr.RV_prim = np.insert(tr.RV_prim, index+1, RV_new, axis=0)
                tr.tag_prim = np.insert(tr.tag_prim, index+1, 11, axis=0)
                break
            # check if the new secondary traj is lower than r_aim
            if (not twisted_fan and
                    np.sign(np.cross(tr.RV_sec[-1, :3], r_aim)[-1]) > 0):
                # if lower, halve the timestep and try once more
                tr.dt1 = tr.dt1/2.
                print('dt1={}'.format(tr.dt1))
                if tr.dt1 < 1e-10:
                    print('dt too small')
                    break
            else:
                # if higher, continue steps along the primary
                RV_old = RV_new

        print('IsAimXY = ', tr.IsAimXY)
        print('IsAimZ = ', tr.IsAimZ)

        if optimize:
            # change UB2 value proportional to dz
            if not tr.IsAimZ:
                dz = r_aim[2]-tr.RV_sec[-1, 2]
                print('UB2 OLD = {:.2f}, z_aim - z = {:.4f} m'
                      .format(UB2, dz))
                UB2 = UB2 - dUB2*dz
                print('UB2 NEW = {:.2f}'.format(UB2))
                attempts_opt += 1
            else:
                break
            # check if there is a loop while finding secondary to aim
            if attempts_opt > 20:
                print('too many attempts B2!')
                break
        else:
            print('B2 was not optimized')
            break
    return tr


# %%
def optimize_A3B3(tr, geom, UA3, UB3, dUA3, dUB3,
                  E, B, dt, target='slit', UA3_max=50., UB3_max=50.,
                  eps_xy=1e-3, eps_z=1e-3):
    ''' get voltages on A3 and B3 plates to get into rs
    '''
    print('\nEb = {}, UA2 = {}'.format(tr.Ebeam, tr.U['A2']))
    print('Target: ' + target)
    if target == 'slit':
        rs = geom.r_dict['slit']
        stop_plane_n = geom.slit_plane_n
    elif target == 'det':
        rs = geom.r_dict['det']
        stop_plane_n = geom.det_plane_n
    elif target == 'A4':
        rs = geom.r_dict['A4']
        stop_plane_n = geom.slit_plane_n

    tr.dt1 = dt
    tr.dt2 = dt
    tmax = 9e-5
    tr.IsAimXY = False
    tr.IsAimZ = False
    RV0 = np.array([tr.RV_sec[0]])

    vltg_fail = False  # flag to track voltage failure
    n_stepsA3 = 0
    while not (tr.IsAimXY and tr.IsAimZ):
        tr.U['A3'], tr.U['B3'] = UA3, UB3
        tr.pass_sec(RV0, rs, E, B, geom,  # stop_plane_n=stop_plane_n,
                    tmax=tmax, eps_xy=eps_xy, eps_z=eps_z)

        drXY = np.linalg.norm(rs[:2]-tr.RV_sec[-1, :2]) * \
            np.sign(np.cross(tr.RV_sec[-1, :2], rs[:2]))
        print('\n UA3 OLD = {:.2f} kV, dr XY = {:.4f} m'.format(UA3, drXY))
        print('IsAimXY = ', tr.IsAimXY)
        # if drXY < 1e-2:
        #     dUA3 = 10.0

        UA3 = UA3 + dUA3*drXY
        print('UA3 NEW = {:.2f} kV'.format(UA3))
        n_stepsA3 += 1

        if abs(UA3) > UA3_max:
            print('ALPHA3 failed, voltage too high')
            vltg_fail = True
            return tr, vltg_fail
        if n_stepsA3 > 100:
            print('ALPHA3 failed, too many steps')
            vltg_fail = True
            return tr, vltg_fail

        # dz = rs[2] - tr.RV_sec[-1, 2]
        # print('\n UB3 OLD = {:.2f} kV, dZ = {:.4f} m'.format(UB3, dz))
        if abs(drXY) < 0.01:
            if tr.IntersectGeometrySec['A3']:
                print('BAD A3!')
                vltg_fail = True
                return tr, vltg_fail
            n_stepsZ = 0
            while not tr.IsAimZ:
                print('pushing Z direction')
                tr.U['A3'], tr.U['B3'] = UA3, UB3
                tr.pass_sec(RV0, rs, E, B, geom,  # stop_plane_n=stop_plane_n,
                            tmax=tmax, eps_xy=eps_xy, eps_z=eps_z)
                # tr.IsAimZ = True  # if you want to skip UB3 calculation
                dz = rs[2] - tr.RV_sec[-1, 2]
                print(' UB3 OLD = {:.2f} kV, dZ = {:.4f} m'
                      .format(UB3, dz))
                print('IsAimXY = ', tr.IsAimXY)
                print('IsAimZ = ', tr.IsAimZ)

                UB3 = UB3 - dUB3*dz
                n_stepsZ += 1
                if abs(UB3) > UB3_max:
                    print('BETA3 failed, voltage too high')
                    vltg_fail = True
                    return tr, vltg_fail
                if n_stepsZ > 100:
                    print('BETA3 failed, too many steps')
                    vltg_fail = True
                    return tr, vltg_fail
                # print('UB3 NEW = {:.2f} kV'.format(UB3))
            n_stepsA3 = 0
            print('n_stepsZ = ', n_stepsZ)
            dz = rs[2] - tr.RV_sec[-1, 2]
            print('UB3 NEW = {:.2f} kV, dZ = {:.4f} m'.format(UB3, dz))

    return tr, vltg_fail


# %%
def optimize_A4(tr, geom, UA4, dUA4,
                E, B, dt, eps_alpha=0.1):
    ''' get voltages on A4 to get proper alpha angle at rs
    '''
    print('\n A4 optimization\n')
    print('\nEb = {}, UA2 = {}'.format(tr.Ebeam, tr.U['A2']))

    rs = geom.r_dict['slit']
    stop_plane_n = geom.slit_plane_n
    alpha_target = geom.angles['an']

    tr.dt1 = dt
    tr.dt2 = dt
    tmax = 9e-5
    # tr.IsAimXY = False
    # tr.IsAimZ = False
    RV0 = np.array([tr.RV_sec[0]])
    V_last = tr.RV_sec[-1][3:]
    alpha, beta = calc_angles(V_last)
    dalpha = alpha_target - alpha
    n_stepsA4 = 0
    while (abs(alpha - alpha_target) > eps_alpha):
        tr.U['A4'] = UA4
        tr.pass_sec(RV0, rs, E, B, geom,  # stop_plane_n=stop_plane_n,
                    tmax=tmax, eps_xy=1e-2, eps_z=1e-2)

        V_last = tr.RV_sec[-1][3:]
        alpha, beta = calc_angles(V_last)
        dalpha = alpha_target - alpha
        print('\n UA4 OLD = {:.2f} kV, dalpha = {:.4f} deg'.format(UA4, dalpha))
        drXY = np.linalg.norm(rs[:2]-tr.RV_sec[-1, :2]) * \
            np.sign(np.cross(tr.RV_sec[-1, :2], rs[:2]))
        dz = rs[2] - tr.RV_sec[-1, 2]
        print('dr XY = {:.4f} m, dz = {:.4f} m'.format(drXY, dz))

        UA4 = UA4 + dUA4*dalpha
        print('UA4 NEW = {:.2f} kV'.format(UA4))
        n_stepsA4 += 1

        if abs(UA4) > 50.:
            print('ALPHA4 failed, voltage too high')
            return tr
        if n_stepsA4 > 100:
            print('ALPHA4 failed, too many steps')
            return tr

    return tr


# %%
def pass_to_slits(tr, dt, E, B, geom, target='slit', timestep_divider=10,
                  no_intersect=True, no_out_of_bounds=True):
    ''' pass trajectories to slits and save secondaries which get into slits
    '''
    tr.dt1 = dt
    tr.dt2 = dt
    k = tr.q / tr.m
    # find the number of slits
    n_slits = geom.slits_edges.shape[0]
    tr.add_slits(n_slits)
    # find slits position
    if target == 'slit':
        r_slits = geom.slits_edges
        rs = geom.r_dict['slit']
        slit_plane_n = geom.slit_plane_n
        slits_spot = geom.slits_spot
    elif target == 'det':
        r_slits = geom.det_edges
        rs = geom.r_dict['det']
        slit_plane_n = geom.det_plane_n
        slits_spot = geom.det_spot

    # pass fan of trajectories
    tr.pass_fan(rs, E, B, geom, stop_plane_n=slit_plane_n,
                eps_xy=1e-3, eps_z=1e-3,
                no_intersect=no_intersect, no_out_of_bounds=no_out_of_bounds)
    # create slits polygon
    ax_index = np.argmax(slit_plane_n)
    slits_spot_flat = np.delete(slits_spot, ax_index, 1)
    slits_spot_poly = path.Path(slits_spot_flat)

    # find which secondaries get into slits spot
    # list of sec trajectories indexes which get into slits spot
    sec_ind = []
    for i in range(1, len(tr.Fan)):
        fan_tr = tr.Fan[i]
        intersect_coords_flat = np.delete(fan_tr[-1, :3], ax_index, 0)
        if slits_spot_poly.contains_point(intersect_coords_flat):
            sec_ind.append(i)
    if len(sec_ind) == 0:
        print('\nNo secondaries go to slit spot')
        return tr

    print('\nStarting precise fan calculation')
    # divide the timestep
    tr.dt1 = dt/timestep_divider
    tr.dt2 = dt
    # number of steps during new fan calculation
    n_steps = timestep_divider * (len(sec_ind) + 1)
    # list for new trajectories
    fan_list = []
    # take the point to start fan calculation
    RV_old = tr.Fan[sec_ind[0]-1][0]
    RV_old = np.array([RV_old])
    RV_new = RV_old

    i_steps = 0
    while i_steps <= n_steps:
        # pass new secondary trajectory
        tr.pass_sec(RV_new, rs, E, B, geom,
                    stop_plane_n=slit_plane_n, tmax=9e-5,
                    eps_xy=1e-3, eps_z=1)
        # make a step on primary trajectory
        r = RV_old[0, :3]
        B_local = return_B(r, B)
        E_local = np.array([0., 0., 0.])
        RV_new = runge_kutt(k, RV_old, tr.dt1, E_local, B_local)
        RV_old = RV_new
        i_steps += 1
        if not (True in tr.IntersectGeometrySec.values() or
                tr.B_out_of_bounds):
            fan_list.append(tr.RV_sec)
    print('\nPrecise fan calculated')

    # choose secondaries which get into slits
    # start slit cycle
    for i_slit in range(n_slits):
        print('\nslit = {}'.format(i_slit+1))
        print('center of the slit = ', r_slits[i_slit, 0, :], '\n')
        # create slit polygon
        slit_flat = np.delete(r_slits[i_slit, 1:, :], ax_index, 1)
        slit_poly = path.Path(slit_flat)
        zones_list = []  # list for ion zones coordinates
        rv_list = []  # list for RV arrays of secondaries
        for fan_tr in fan_list:
            # get last coordinates of the secondary trajectory
            intersect_coords_flat = np.delete(fan_tr[-1, :3], ax_index, 0)
            if slit_poly.contains_point(intersect_coords_flat):
                print('slit {} ok!\n'.format(i_slit+1))
                rv_list.append(fan_tr)
                zones_list.append(fan_tr[0, :3])

        tr.RV_sec_toslits[i_slit] = rv_list
        tr.ion_zones[i_slit] = np.array(zones_list)
    tr.fan_to_slits = fan_list

    return tr


# %%
@numba.jit()
def translate(input_array, xyz):
    '''
    move the vector in space
    :param xyz: 3 component vector
    :return: translated input_array
    '''
    if input_array is not None:
        input_array += np.array(xyz)

    return input_array


@numba.jit()
def rot_mx(axis=(1, 0, 0), deg=0):
    '''
    function calculates rotation matrix
    :return: rotation matrix
    '''
    n = axis
    ca = np.cos(np.radians(deg))
    sa = np.sin(np.radians(deg))
    R = np.array([[n[0]**2*(1-ca)+ca, n[0]*n[1]*(1-ca)-n[2]*sa,
                   n[0]*n[2]*(1-ca)+n[1]*sa],

                  [n[1]*n[0]*(1-ca)+n[2]*sa, n[1]**2*(1-ca)+ca,
                   n[1]*n[2]*(1-ca)-n[0]*sa],

                  [n[2]*n[0]*(1-ca)-n[1]*sa, n[2]*n[1]*(1-ca)+n[0]*sa,
                   n[2]**2*(1-ca)+ca]])
    return R


@numba.jit()
def rotate(input_array, axis=(1, 0, 0), deg=0.):
    '''
    rotate vector around given axis by deg degrees
    :param axis: axis of rotation
    :param deg: angle in degrees
    :return: rotated input_array
    '''
    if input_array is not None:
        R = rot_mx(axis, deg)
        input_array = np.dot(input_array, R.T)
    return input_array


@numba.jit()
def rotate3(input_array, plates_angles, beamline_angles, inverse=False):
    '''
    rotate vector in 3 dimentions
    plates_angles - angles of the plates
    beamline_angles - angles of the beamline axis, rotation on gamma angle
    '''
    alpha, beta, gamma = plates_angles
    axis = calc_vector(1, beamline_angles[0], beamline_angles[1])

    if inverse:
        rotated_array = rotate(input_array, axis=axis, deg=-gamma)
        rotated_array = rotate(rotated_array, axis=(0, 1, 0), deg=-beta)
        rotated_array = rotate(rotated_array, axis=(0, 0, 1), deg=-alpha)
    else:
        rotated_array = rotate(input_array, axis=(0, 0, 1), deg=alpha)
        rotated_array = rotate(rotated_array, axis=(0, 1, 0), deg=beta)
        rotated_array = rotate(rotated_array, axis=axis, deg=gamma)
    return rotated_array


@numba.jit()
def reflect(input_array):
    '''
    reflect vector to the HIBP cross section in TJ2
    '''
    x1, y1, z1 = input_array
    fi = np.arctan(z1/x1) * 180./np.pi
    x2, y2, z2 = np.sqrt(x1**2 + z1**2) - 1.5, y1, 0.0

    reflected_array = rotate(np.array([x2, y2, z2]), axis=(0, 0, 1), deg=-4*fi)
    reflected_array[0] += 1.5
    return reflected_array


# %% Intersection check functions
def line_plane_intersect(planeNormal, planePoint, rayDirection,
                         rayPoint, eps=1e-6):
    ''' function returns intersection point between plane and ray
    '''
    ndotu = np.dot(planeNormal, rayDirection)
    if abs(ndotu) < eps:
        # print('no intersection or line is within plane')
        return np.full_like(planeNormal, np.nan)
    else:
        w = rayPoint - planePoint
        si = -np.dot(planeNormal, w) / ndotu
        Psi = w + si * rayDirection + planePoint
        return Psi


def is_between(A, B, C, eps=1e-6):
    ''' function returns True if point C is on the segment AB
    (between A and B)'''
    if np.isnan(C).any():
        return False
    # check if the points are on the same line
    crossprod = np.cross(B-A, C-A)
    if np.linalg.norm(crossprod) > eps:
        return False
    # check if the point is between
    dotprod = np.dot(B-A, C-A)
    if dotprod < 0 or dotprod > np.linalg.norm(B-A)**2:
        return False
    return True


def segm_intersect(A, B, C, D):  # doesn't work with collinear case
    ''' function returns true if line segments AB and CD intersect
    '''
    def order(A, B, C):
        ''' If counterclockwise return True
            If clockwise return False '''
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    # Return true if line segments AB and CD intersect
    return order(A, C, D) != order(B, C, D) and \
        order(A, B, C) != order(A, B, D)


def segm_poly_intersect(polygon_coords, segment_coords):
    ''' check segment and polygon intersection'''
    polygon_normal = np.cross(polygon_coords[2, 0:3]-polygon_coords[0, 0:3],
                              polygon_coords[1, 0:3]-polygon_coords[0, 0:3])
    polygon_normal = polygon_normal/np.linalg.norm(polygon_normal)

    intersect_coords = line_plane_intersect(polygon_normal,
                                            polygon_coords[2, 0:3],
                                            segment_coords[1, 0:3] -
                                            segment_coords[0, 0:3],
                                            segment_coords[0, 0:3])
    if np.isnan(intersect_coords).any():
        return False
    else:
        i = np.argmax(abs(polygon_normal))
        polygon_coords_flat = np.delete(polygon_coords, i, 1)
        intersect_coords_flat = np.delete(intersect_coords, i, 0)
        p = path.Path(polygon_coords_flat)
        return p.contains_point(intersect_coords_flat) and \
            is_between(segment_coords[0, 0:3], segment_coords[1, 0:3],
                       intersect_coords)

    # check projections on XY and XZ planes
    # pXY = path.Path(polygon_coords[:, [0, 1]])  # XY plane
    # pXZ = path.Path(polygon_coords[:, [0, 2]])  # XZ plane
    # return pXY.contains_point(intersect_coords[[0, 1]]) and \
    #     pXZ.contains_point(intersect_coords[[0, 2]]) and \
    #         is_between(segment_coords[0, 0:3], segment_coords[1, 0:3],
    #                   intersect_coords)


# %%
def plate_flags(range_x, range_y, range_z, U,
                plts_geom, gamma, plts_center):

    length, width, thick, gap = plts_geom

    # Geometry rotated in system based on central point between plates
    # upper plate
    UP1 = np.array([-length/2., gap/2. + thick, width/2.])
    UP2 = np.array([-length/2., gap/2. + thick, -width/2.])
    UP3 = np.array([length/2., gap/2. + thick, -width/2.])
    UP4 = np.array([length/2., gap/2. + thick, width/2.])
    UP5 = np.array([-length/2., gap/2., width/2.])
    UP6 = np.array([-length/2., gap/2., -width/2.])
    UP7 = np.array([length/2., gap/2., -width/2.])
    UP8 = np.array([length/2., gap/2., width/2.])
    UP_full = np.array([UP1, UP2, UP3, UP4, UP5, UP6, UP7, UP8])
    UP_rotated = UP_full.copy()
    for i in range(UP_full.shape[0]):
        UP_rotated[i, :] = rotate(UP_rotated[i, :], axis=(1, 0, 0), deg=gamma)
        # shift coords center
        UP_rotated[i, :] += plts_center

    # lower plate
    LP1 = np.array([-length/2., -gap/2. - thick, width/2.])
    LP2 = np.array([-length/2., -gap/2. - thick, -width/2.])
    LP3 = np.array([length/2., -gap/2. - thick, -width/2.])
    LP4 = np.array([length/2., -gap/2. - thick, width/2.])
    LP5 = np.array([-length/2., -gap/2., width/2.])
    LP6 = np.array([-length/2., -gap/2., -width/2.])
    LP7 = np.array([length/2., -gap/2., -width/2.])
    LP8 = np.array([length/2., -gap/2., width/2.])
    LP_full = np.array([LP1, LP2, LP3, LP4, LP5, LP6, LP7, LP8])
    LP_rotated = LP_full.copy()
    for i in range(LP_full.shape[0]):
        LP_rotated[i, :] = rotate(LP_rotated[i, :], axis=(1, 0, 0), deg=gamma)
        # shift coords center
        LP_rotated[i, :] += plts_center

    # Find coords of 'cubes' containing each plate
    upper_cube = np.array([np.min(UP_rotated, axis=0),
                           np.max(UP_rotated, axis=0)])
    lower_cube = np.array([np.min(LP_rotated, axis=0),
                           np.max(LP_rotated, axis=0)])

    # create mask for plates
    upper_plate_flag = np.full_like(U, False, dtype=bool)
    lower_plate_flag = np.full_like(U, False, dtype=bool)
    for i in range(range_x.shape[0]):
        for j in range(range_y.shape[0]):
            for k in range(range_z.shape[0]):
                x = range_x[i]
                y = range_y[j]
                z = range_z[k]
                # check upper cube
                if (x >= upper_cube[0, 0]) and (x <= upper_cube[1, 0]) and \
                   (y >= upper_cube[0, 1]) and (y <= upper_cube[1, 1]) and \
                   (z >= upper_cube[0, 2]) and (z <= upper_cube[1, 2]):
                    r = np.array([x, y, z]) - plts_center
                    # inverse rotation
                    r_rot = rotate(r, axis=(1, 0, 0), deg=-gamma)
                    # define masks for upper and lower plates
                    upper_plate_flag[i, j, k] = (r_rot[0] >= -length/2.) and \
                        (r_rot[0] <= length/2.) and (r_rot[2] >= -width/2.) and \
                        (r_rot[2] <= width/2.) and (r_rot[1] >= gap/2.) and \
                        (r_rot[1] <= gap/2. + thick)
                # check lower cube
                if (x >= lower_cube[0, 0]) and (x <= lower_cube[1, 0]) and \
                   (y >= lower_cube[0, 1]) and (y <= lower_cube[1, 1]) and \
                   (z >= lower_cube[0, 2]) and (z <= lower_cube[1, 2]):
                    r = np.array([x, y, z]) - plts_center
                    # inverse rotation
                    r_rot = rotate(r, axis=(1, 0, 0), deg=-gamma)
                    # define masks for upper and lower plates
                    lower_plate_flag[i, j, k] = (r_rot[0] >= -length/2.) and \
                        (r_rot[0] <= length/2.) and (r_rot[2] >= -width/2.) and \
                        (r_rot[2] <= width/2.) and \
                        (r_rot[1] >= -gap/2. - thick) and \
                        (r_rot[1] <= -gap/2.)

    return UP_rotated, LP_rotated, upper_plate_flag, lower_plate_flag


def return_E(r, Ein, U, geom):
    '''
    take dot and try to interpolate electiric fields in it
    return: interpolated Electric field
    :param Ein: dict of interpolants for Ex, Ey, Ez
    '''
    Etotal = np.zeros(3)
    # do not check plates while particle is in plasma
    if r[0] < geom.r_dict['aim'][0]-0.05 and r[1] < geom.r_dict['port'][1]:
        return Etotal
    # go through all the plates
    for key in geom.plates_edges.keys():
        # shift the center of coord system
        r_new = r - geom.r_dict[key]
        # get angles
        angles = copy.deepcopy(geom.angles[key])
        beamline_angles = copy.deepcopy(geom.angles[key])
        # change alpha angle of the analyzer
        if key == 'an':
            theta_an = geom.an_params[4]
            angles[0] = angles[0] - theta_an
        # rotate point to the coord system of plates
        r_new = rotate3(r_new, angles, beamline_angles, inverse=True)
        # interpolate Electric field
        Etemp = np.zeros(3)
        try:
            Etemp[0] = Ein[key][0](r_new) * U[key]
            Etemp[1] = Ein[key][1](r_new) * U[key]
            Etemp[2] = Ein[key][2](r_new) * U[key]
            # rotate Etemp
            Etemp = rotate3(Etemp, angles, beamline_angles, inverse=False)
            # add the result to total E field
            Etotal += Etemp
        except (ValueError, IndexError):
            continue
    return Etotal


def return_B(r, Bin):
    Bx_interp, By_interp, Bz_interp = Bin[0], Bin[1], Bin[2]
    Bout = np.c_[Bx_interp(r), By_interp(r), Bz_interp(r)]
    return Bout


def save_E(beamline, plts_name, Ex, Ey, Ez, gamma, geom,
           domain, an_params, plate1, plate2, dirname='elecfield'):
    '''
    save Ex, Ey, Ez arrays to file
    '''
    dirname = dirname + '/' + beamline

    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname, 0o700)
            print("Directory ", dirname, " created ")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    fname = plts_name + '_geometry.dat'
    # erases data from file before writing
    open(dirname + '/' + fname, 'w').close()
    with open(dirname + '/' + fname, 'w') as myfile:
        myfile.write(np.array2string(geom)[1:-1] +
                     ' # plate\'s length, width, thic and gap\n')
        myfile.write(str(gamma) +
                     ' # plate\'s gamma angle\n')
        myfile.write(np.array2string(domain, max_line_width=200)[1:-1] +
                     ' # xmin, xmax, ymin, ymax, zmin, zmax, delta\n')
        if plts_name == 'an':
            myfile.write(np.array2string(an_params, max_line_width=200)[1:-1] +
                         ' # n_slits, slit_dist, slit_w, G, theta, XD, YD1, YD2\n')
        for i in range(plate1.shape[0]):
            myfile.write(np.array2string(plate1[i], precision=4)[1:-1] +
                         ' # 1st plate rotated\n')
        for i in range(plate2.shape[0]):
            myfile.write(np.array2string(plate2[i], precision=4)[1:-1] +
                         ' # 2nd plate rotated\n')

    np.save(dirname + '/' + plts_name + '_Ex', Ex)
    np.save(dirname + '/' + plts_name + '_Ey', Ey)
    np.save(dirname + '/' + plts_name + '_Ez', Ez)

    print('Electric field saved, ' + plts_name + '\n')


def read_E(beamline, geom, dirname='elecfield'):
    '''
    read plate's shape and angle parametres along with electric field values
    from provided file (should lbe in the same directory)
    return: intrepolator function for electric field
    :param fname: filename
    :param r_dict: dict (key: name of plate,
                         value: coordinate array of plates centre)
    '''
    r_dict = geom.r_dict

    E = {}
    edges_dict = {}
    dirname = dirname + '/' + beamline

    # list of all *.dat files
    file_list = [file for file in os.listdir(dirname) if file.endswith('dat')]
    # push analyzer to the end of the list
    file_list.sort(key=lambda s: s[1:])  # A3, B3, A4, an
    # file_list.sort(key=lambda s: s.startswith('an'))

    for filename in file_list:
        plts_name = filename[0:2]
        r_new = r_dict[plts_name]
        angles = copy.deepcopy(geom.angles[plts_name])
        beamline_angles = copy.deepcopy(geom.angles[plts_name])
        print('position', r_new)

        edges_list = []
        print('\n Reading geometry {} ...'.format(plts_name))

        with open(dirname + '/' + filename, 'r') as f:
            geometry = [float(i) for i in f.readline().split()[0:4]]
            gamma = float(f.readline().split()[0])
            # xmin, xmax, ymin, ymax, zmin, zmax, delta
            domain = [float(i) for i in f.readline().split()[0:7]]
            if plts_name == 'an':
                an_params = [float(i) for i in f.readline().split()[0:8]]
                geom.an_params = np.array(an_params)
                theta_an = geom.an_params[4]  # analyzer entrance angle
                angles[0] = angles[0] - theta_an
            for line in f:
                edges_list.append([float(i) for i in line.split()[0:3]])

        edges_list = np.array(edges_list)
        # rotate plates edges
        for i in range(edges_list.shape[0]):
            edges_list[i, :] = rotate3(edges_list[i, :],
                                       angles, beamline_angles)
        # shift coords center and put into a dictionary
        edges_dict[plts_name] = np.array([edges_list[0:4, :] + r_new,
                                          edges_list[4:, :] + r_new])

        Ex = np.load(dirname + '/' + plts_name + '_Ex.npy')
        Ey = np.load(dirname + '/' + plts_name + '_Ey.npy')
        Ez = np.load(dirname + '/' + plts_name + '_Ez.npy')

        x = np.arange(domain[0], domain[1], domain[6])  # + r_new[0]
        y = np.arange(domain[2], domain[3], domain[6])  # + r_new[1]
        z = np.arange(domain[4], domain[5], domain[6])  # + r_new[2]

        # make interpolation for Ex, Ey, Ez
        Ex_interp = RegularGridInterpolator((x, y, z), Ex)
        Ey_interp = RegularGridInterpolator((x, y, z), Ey)
        Ez_interp = RegularGridInterpolator((x, y, z), Ez)
        E_read = [Ex_interp, Ey_interp, Ez_interp]

        E[plts_name] = E_read

    return E, edges_dict


def read_B(config, dirname='tj2lib', interp=True):
    '''
    read magnetic field values from provided file (should be in
                                                   the same directory)
    return: list of intrepolator functions for Bx, By, Bz
    :param dirname: name of directory with magfield dats
    '''
    print('\n Reading Magnetic field')
    print('Configuration ', config)
    # read precalculated magnetic field from binary *dat file
    # x, y, z, Bx, By, Bz, rho
    B_read = np.fromfile(dirname + '//' + config + '_xyz_Brho.dat')
    # reshape into [Npoints, 7] array
    B_read = B_read.reshape([int(B_read.shape[0] / 7), 7])

    # convert B from TJ-II coord system to HIBP coord system
    # HIBP2 toroidal angle
    theta = 38.172
    hibp2_tor_angle = 180. + theta
    for i in range(B_read.shape[0]):
        # rotate coords
        # r_tj2 = B_read[i, 0:3]
        # r_hibp = rotate(r_tj2, axis=(0, 0, 1), deg=-hibp2_tor_angle)
        # r_hibp = rotate(r_hibp, axis=(1, 0, 0), deg=-90.)
        # B_read[i, 0:3] = r_hibp
        # rotate and inverse B
        b_tj2 = -B_read[i, 3:6]
        b_hibp = rotate(b_tj2, axis=(0, 0, 1), deg=-hibp2_tor_angle)
        b_hibp = rotate(b_hibp, axis=(1, 0, 0), deg=-90.)
        B_read[i, 3:6] = b_hibp

    with open(dirname + '/' + '3d_b_geometry.dat', 'r') as f:
        volume_corner1 = [float(i) for i in f.readline().split()[0:3]]
        volume_corner2 = [float(i) for i in f.readline().split()[0:3]]
        resolution = float(f.readline().split()[0])

    # create grid of points
    grid = np.mgrid[volume_corner1[0]:volume_corner2[0]:resolution,
                    volume_corner1[1]:volume_corner2[1]:resolution,
                    volume_corner1[2]:volume_corner2[2]:resolution]

    B = B_read[:, 3:6]
    rho = B_read[:, 6]

    # plot B stream
    hbplot.plot_B_stream(B, volume_corner1, volume_corner2, resolution, grid,
                         plot_sep=True)

    x = np.arange(volume_corner1[0], volume_corner2[0], resolution)
    y = np.arange(volume_corner1[1], volume_corner2[1], resolution)
    z = np.arange(volume_corner1[2], volume_corner2[2], resolution)
    Bx = B[:, 0].reshape(grid.shape[1:])
    By = B[:, 1].reshape(grid.shape[1:])
    Bz = B[:, 2].reshape(grid.shape[1:])
    rho = rho.reshape(grid.shape[1:])
    if interp:
        # make an interpolation of B
        Bx_interp = RegularGridInterpolator((x, y, z), Bx)
        By_interp = RegularGridInterpolator((x, y, z), By)
        Bz_interp = RegularGridInterpolator((x, y, z), Bz)
        print('Interpolants for magnetic field created')
        B_list = [Bx_interp, By_interp, Bz_interp]
        rho_interp = RegularGridInterpolator((x, y, z), rho)
        print('Interpolant for rho created')
    else:
        B_list = [Bx, By, Bz]
        rho_interp = rho

    return B_list, rho_interp


# %%
def save_traj_list(traj_list, config, r_aim, dirname='output'):
    '''
    Save list of Traj objects to pickle file
    :param traj_list: list of trajectories
    '''

    if len(traj_list) == 0:
        print('traj_list empty!')
        return

    Ebeam_list = []
    UA2_list = []

    for traj in traj_list:
        Ebeam_list.append(traj.Ebeam)
        UA2_list.append(traj.U['A2'])

    dirname = dirname + '/' + config

    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname, 0o700)
            print("Directory ", dirname, " created ")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    fname = dirname + '/' + 'E{}-{}'.format(int(min(Ebeam_list)),
                                            int(max(Ebeam_list))) + \
        '_UA2{}-{}'.format(int(min(UA2_list)), int(max(UA2_list))) + \
        '_alpha{}_beta{}'.format(int(round(traj.alpha)),
                                 int(round(traj.beta))) +\
        '_x{}y{}z{}.pkl'.format(int(r_aim[0]*100), int(r_aim[1]*100),
                                int(r_aim[2]*100))

    with open(fname, 'wb') as f:
        pc.dump(traj_list, f, -1)

    print('\nSAVED LIST: \n' + fname)


# %%
def read_traj_list(fname, dirname='output'):
    '''
    import list of Traj objects from .pkl file
    '''
    with open(dirname + '/' + fname, 'rb') as f:
        traj_list = pc.load(f)
    return traj_list


# %%
def save_traj2dat(traj_list, save_fan=False, dirname='output/',
                  fmt='%.2f', delimiter=' '):
    '''
    save list of trajectories to *.dat files for CATIA plot
    '''
    for tr in traj_list:
        # save primary
        fname = dirname + 'E{:.0f}_U{:.0f}_prim.dat'.format(tr.Ebeam, tr.U['A2'])
        np.savetxt(fname, tr.RV_prim[:, 0:3]*1000,
                   fmt=fmt, delimiter=delimiter)  # [mm]
        # save secondary
        fname = dirname + 'E{:.0f}_U{:.0f}_sec.dat'.format(tr.Ebeam, tr.U['A2'])
        np.savetxt(fname, tr.RV_sec[:, 0:3]*1000,
                   fmt=fmt, delimiter=delimiter)


# %%
def save_radref(traj_list, Ebeam, rho_interp, dirname='radref/',
                fmt='%.3f', delimiter=' '):
    '''
    save radial reference for a certain energy
    '''
    fname = dirname + 'radref_E{}.dat'.format(int(Ebeam))
    # E, UA2, rho, x_reflected, y_reflected, x, y, z, UB2, UA3, UB3
    radref = np.empty([0, 11])
    for tr in traj_list:
        if tr.Ebeam == Ebeam:
            x, y, z = tr.RV_sec[0, :3]
            xyz_refl = reflect(tr.RV_sec[0, :3])
            radref_temp = np.array([Ebeam, tr.U['A2'],
                                    rho_interp(tr.RV_sec[0, :3])[0],
                                    xyz_refl[0], xyz_refl[1], x, y, z,
                                    tr.U['B2'], tr.U['A3'], tr.U['B3']])
            radref = np.vstack([radref, radref_temp])
    # set rho<0 at HFS
    distances = radref[:-1, 2] - radref[1:, 2]
    index = np.argwhere(distances < 0)[0][0]
    radref[:index+1, 2] = radref[:index+1, 2] * np.sign(radref[0, 1])
    radref[index+1:, 2] = radref[index+1:, 2] * np.sign(radref[-1, 1])
    # save to file
    np.savetxt(fname, radref, fmt=fmt, delimiter=delimiter)
    print('radref E={} saved'.format(int(Ebeam)))


# %%
def save_png(fig, name, save_dir='output'):
    """
    Saves picture as name.png
    Args:
    :fig - array of figures to save
    :name - array of picture names
    :save_dir - directory used to store results
    """

    # check wether directory exist and if not - create one
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        print('LOG: {} directory created'.format(save_dir))
    print('LOG: Saving pictures to {}'.format(save_dir+'/'))
    for fig, name in zip(fig, name):
        # save fig with tight layout
        fig_savename = str(name + '.png')
        fig.savefig(save_dir + '/' + fig_savename, bbox_inches='tight')
        print('LOG: Figure ' + fig_savename + ' saved')
