import numpy as np
import hibplib as hb


def define_geometry(config, analyzer=1):
    geom = hb.Geometry()

    # plasma parameters
    geom.R = 1.5  # tokamak major radius [m]
    geom.r_plasma = 0.4  # plasma minor radius [m]
    geom.elon = 1.  # plasma elongation

    # PRIMARY beamline geometry
    # alpha and beta angles of the PRIMARY beamline [deg]
    alpha_prim = 74.217
    beta_prim = -11.712
    gamma_prim = 0.
    prim_angles = {'r0': np.array([alpha_prim, beta_prim, gamma_prim]),
                   'A1': np.array([alpha_prim, beta_prim, gamma_prim]),
                   'B1': np.array([alpha_prim, beta_prim, gamma_prim]),
                   'B2': np.array([alpha_prim, beta_prim, gamma_prim]),
                   'A2': np.array([alpha_prim, beta_prim, gamma_prim])}
    geom.angles.update(prim_angles)

    # coordinates of the injection port [m]
    xport = 1.3268  # 1.5 - 0.1541
    yport = 0.853
    zport = 0.032
    geom.r_dict['port'] = np.array([xport, yport, zport])

    # distance from the injection port to the Alpha2 plates
    dist_A2 = 0.243  # 0.552  # 0.2  # [m]
    # distance from Alpha2 plates to the Beta2 plates
    dist_B2 = 0.188  # [m]
    # distance from Beta2 plates to the Beta1 plates
    dist_B1 = 0.416
    # distance from Beta1 plates to the Alpha1 plates
    dist_A1 = 0.176
    # distance from Alpha1 plates to the initial point of the traj [m]
    dist_r0 = 0.2

    # coordinates of the center of the ALPHA2 plates
    geom.add_coords('A2', 'port', dist_A2, geom.angles['A2'])
    # coordinates of the center of the BETA2 plates
    geom.add_coords('B2', 'A2', dist_B2, geom.angles['B2'])
    # coordinates of the center of the BETA2 plates
    geom.add_coords('B1', 'B2', dist_B1, geom.angles['B1'])
    # coordinates of the center of the BETA2 plates
    geom.add_coords('A1', 'B1', dist_A1, geom.angles['A1'])
    # coordinates of the initial point of the trajectory [m]
    geom.add_coords('r0', 'A1', dist_r0, geom.angles['r0'])

    # AIM position (BEFORE the Secondary beamline) [m]
    xaim = 1.923  # 2.408  # 1.9
    yaim = -0.456
    zaim = 0.0428  # -0.116  # 0.0
    r_aim = np.array([xaim, yaim, zaim])
    geom.r_dict['aim'] = r_aim

    # SECONDARY beamline geometry
    # alpha and beta angles of the SECONDARY beamline [deg]
    alpha_sec = 0.
    beta_sec = 15.5
    gamma_sec = 17.
    sec_angles = {'aim1': np.array([alpha_sec, beta_sec, gamma_sec]),
                  'A3': np.array([alpha_sec, beta_sec, gamma_sec]),
                  'B3': np.array([alpha_sec, 12., gamma_sec]),
                  'A4': np.array([alpha_sec, 4., 25.]),
                  'B4': np.array([alpha_sec, 4., 25.]),
                  'an': np.array([alpha_sec, 4., 25.])}
    geom.angles.update(sec_angles)

    # distance from r_aim to the Alpha3 center
    dist_A3 = 0.9343  # 0.4  # 0.3  # 1/2 of plates length
    # distance from Alpha3 to the Beta3 center
    dist_B3 = 0.31  # + 0.6
    # from Beta3 to Beta4
    dist_B4 = 0.66
    # from Beta3 to Aalpha4
    dist_A4 = 1.0
    # distance from Beta3 to the entrance slit of the analyzer
    dist_s = 1.3

    # coordinates of the center of the new aim point
    geom.add_coords('aim1', 'aim', dist_A3-0.1, geom.angles['A3'])
    # coordinates of the center of the ALPHA3 plates
    geom.add_coords('A3', 'aim', dist_A3, geom.angles['A3'])
    # coordinates of the center of the BETA3 plates
    geom.add_coords('B3', 'A3', dist_B3, geom.angles['B3'])
    # coordinates of the center of the BETA4 plates
    geom.add_coords('B4', 'B3', dist_B4, geom.angles['B4'])
    # coordinates of the center of the ALPHA4 plates
    geom.add_coords('A4', 'B3', dist_A4, geom.angles['A4'])
    # Coordinates of the CENTRAL slit
    geom.add_coords('slit', 'B3', dist_s, geom.angles['an'])
    # Coordinates of the ANALYZER
    geom.add_coords('an', 'B3', dist_s, geom.angles['an'])

    # print info
    print('\nDefining geometry for Analyzer #{}'.format(analyzer))
    print('\nPrimary beamline angles: ', geom.angles['r0'])
    print('Secondary beamline angles: ', geom.angles['A3'])
    print('r0 = ', np.round(geom.r_dict['r0'], 3))
    print('r_aim = ', r_aim)
    print('r_slit = ', np.round(geom.r_dict['slit'], 3))

    # TJ-II GEOMETRY
    # chamber entrance and exit coordinates
    geom.chamb_ent = [(0.8, 0.5805), (1.1, 0.5805),
                      (1.43, 0.5805), (1.8, 0.5805)]
    geom.chamb_ext = [(1.89, -0.28), (1.89, 0.0),
                      (1.89, -0.66), (1.89, -0.9),
                      (1.58, -0.52), (1.58, -0.9)]
    geom.chamb = [(1.415, -0.07136), (1.4007, -0.04931),
                  (1.4007, -0.04931), (1.3919, -0.02453),
                  (1.3919, -0.02453), (1.389, 0.00162),
                  (1.389, 0.00162), (1.3925, 0.0277),
                  (1.3925, 0.0277), (1.4021, 0.05218),
                  (1.4021, 0.05218), (1.416, 0.07456),
                  (1.416, 0.07456), (1.4302, 0.09681),
                  (1.4302, 0.09681), (1.4443, 0.11909)]

    # Camera contour
    geom.camera = np.loadtxt('TJII_camera.dat')
    # Separatrix contour
    geom.sep = np.loadtxt('configs//' + config + '.txt')

    return geom
