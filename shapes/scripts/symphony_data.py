import symlib
import matplotlib.pyplot as plt
import numpy as np
import pynbody
#import pyEXP
from scipy.linalg import norm
import sys
from shape_helpers import profiles_multipanel, save_properties 

sim_dir = "/mnt/home/nico/ceph/symphony/MWest/Halo004/output/"
#sim_dir = "/mnt/home/nico/ceph/symphony/SymphonyMW/Halo023/output/"
snap_init = int(sys.argv[1])
snap_final = int(sys.argv[2])

def density_slice(image, fig_name):
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    img = plt.imshow(np.log10(image), origin='lower', extent=[-500, 500, -500, 500], cmap='twilight', vmin=5, vmax=9)
    plt.colorbar(img)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    shapes = []
    profiles = []
    for i in range(snap_init, snap_final, 1):
        # Load sim
        f = pynbody.load(sim_dir + "snapshot_{:03d}".format(i))
        f.physical_units()
        # Select zoom region
        zoomin_filter = pynbody.filt.BandPass('mass', '400000 Msol', '500000 Msol')
        fzoom = f.dm[zoomin_filter] 
        # Center halo
        pynbody.analysis.halo.center(fzoom, mode='ssc', retcen=False, vel=True)
        # Take 500 kpc radius
        radius_filter = pynbody.filt.Annulus('0 kpc', '500 kpc')
        fhalo_zoom = fzoom[radius_filter]
        #im = pynbody.plot.image(fhalo_zoom, width="0.6 Mpc", units="Msol kpc^-2", cmap="twilight");
        #fig_name = "sym_dslice_halo_023_snap_{:03d}".format(i)
        #density_slice(im, fig_name)
        shape = pynbody.analysis.halo.halo_shape(fhalo_zoom, N=10, rin=0.1, rout=200, bins='equal')
        rbins = shape[0]
        profile = pynbody.analysis.profile.Profile(fhalo_zoom, ndim=3, bins=rbins)
        #profiles.append(profile)

        #fig_name = "sym_profiles_halo_023_snap_{:03d}".format(i)
        file_name = "mwest_halo_004_snap_{:03d}".format(i)
        #profiles_multipanel(shape, profiles, fig_name)
        save_properties(shape, profile, file_name)
        #p.save("profiles_halo_023", profiles)
        #p.save("shapes_halo_023", shapes)
