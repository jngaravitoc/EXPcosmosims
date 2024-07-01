import numpy as np
import k3d
import pickle
import EXPtools
import sys

def load_fields(f0, fall):
    monopole = np.load(f0)
    full_exp = np.load(fall)
    return monopole, full_exp

def load_orbit(orbit_file):
    orbit = np.loadtxt(orbit_file)
    return orbit


def make_3d_render(field, orbit, halo_name):
    rend3D =  EXPtools.visuals.field3Drender(field, 
                                             contour_ranges=[[-1, 1],[-0.8, 0.8]],
                                             size=[150, 150, 150],
                                             contour_alphas=[2, 1],
                                             orbits=[np.array([orbit[:,2], orbit[:,0], orbit[:,1]]).T],
                                             orbits_names=[halo_name])
    
    with open('{}_3D_render.html'.format(halo_name) ,'w') as fp:
       fp.write(rend3D.get_snapshot())

    return 0


if __name__ == "__main__":
    fields_path = sys.argv[1]
    orbit_path = sys.argv[2]
    halo_id = int(sys.argv[3])
    f0 = fields_path + "/HaloHalo{:03d}_dens0_3D.npy".format(halo_id)
    fall = fields_path + "/HaloHalo{:03d}_dens_3D.npy".format(halo_id)
    forbit = orbit_path + "/HaloHalo{:03d}_orbit.txt".format(halo_id) 
    mon, full = load_fields(f0, fall)
    orbit = load_orbit(forbit)
    make_3d_render(np.array([full/mon]), orbit,
            halo_name="Halo{:03d}".format(halo_id)) 
