import numpy as np
#mport symlib 
import matplotlib.pyplot as plt
#rom tqdm import trange
#from matplotlib.colors import SymLogNorm, LogNorm
#plt.style.use("./vis/matplotlib.mplstyle")
import nba
import os
import pynbody
from data_paths import SNAPSHOTS_DIR
from density_projections import density_projections

def return_bound_particle_ids(p, subhalo_index, E_key='E_sph'):
    is_bound = (p[subhalo_index][E_key] < 0)
    return p[subhalo_index]['id'][is_bound]

def indices_to_keep(particle_ids_host, particle_id_tobeRemoved):
    """
    Return integer indices (into particle_ids_host) for elements NOT in particle_id_tobeRemoved.
    Preserves the order of particle_ids_host. Safe: avoids out-of-bounds when using searchsorted.
    """
    host = np.asarray(particle_ids_host)
    removed = np.asarray(particle_id_tobeRemoved)

    if host.size == 0:
        return np.array([], dtype=int)
    if removed.size == 0:
        return np.arange(host.size, dtype=int)

    # ensure comparable dtypes
    if host.dtype != removed.dtype:
        try:
            removed = removed.astype(host.dtype, copy=False)
        except Exception:
            host = host.astype(removed.dtype, copy=False)

    # unique+sort the removed list (smaller & faster comparisons)
    removed_sorted = np.sort(np.unique(removed))

    # binary-search each host id in the small removed_sorted array
    idx = np.searchsorted(removed_sorted, host)

    # SAFE check: only index removed_sorted where idx is in-boundsl
    found = np.zeros(host.shape, dtype=bool)
    mask_valid = idx < removed_sorted.size
    if mask_valid.any():
        found[mask_valid] = removed_sorted[idx[mask_valid]] == host[mask_valid]

    keep_mask = ~found
    return np.nonzero(keep_mask)[0]


def get_HR_particles(snap_dir):  
    snap = pynbody.snapshot.gadget.GadgetSnap(snap_dir)
    min_pm = np.min(snap['mass'])
    HR_filt = pynbody.filt.LowPass('mass', min_pm+0.1)
    HR_snap = snap[HR_filt]
    pynbody.analysis.halo.center(HR_snap)
    assert snap.header.HubbleParam == 0.7
    return HR_snap

def test_get_HR_particles(snap_dir):
    snap = pynbody.snapshot.gadget.GadgetSnap(snap_dir)
    HR_mask = np.where(snap['mass']==min_pm)
    min_pm = np.min(snap['mass'])
    pos_halo = snap['pos'][HR_mask] 
    pos_halo_filter = get_HR_particles(snap_dir)
    assert len(pos_halo) == pos_halo_filer['pos']
    
def density_plot(snap):
    fig, ax = plt.subplots(1, 1)
    imag = pynbody.plot.image(snap, width="300 kpc", cmap='Greys',  subplot=ax)
    plt.savefig('test_plot.png')
    plt.close()


def compute_density_profile(snap):
    profiles = pynbody.analysis.profile.Profile(snap, ndim=3)
    R = profiles['rbins'].in_units('kpc')
    rho = profiles['density'].in_units('Msol kpc^-3')
    return R, rho

def plot_density_profile(R, profile):
    plt.semilogy(R, R**2*profile)
    plt.savefig("test_density_profile.png", bbox_inches='tight')
    plt.close()

def compute_r200c(rbins, density_profile, rho_crit):
    rho200 = 200*rho_crit
    r200c = np.argmin(density_profile - rho200)
    return rbins[r200c]


if __name__ == "__main__":
    r200c = np.zeros(243)
    for s in range(243):
        HALO = "MWest/Halo004/output/snapshot_{:03d}".format(s)
        snap_dir = os.path.join(SNAPSHOTS_DIR, HALO)
        snap_HR = get_HR_particles(snap_dir)
        snap_HR.physical_units()
        #redshift = snap.header.redshift
        rho_crit = pynbody.analysis.cosmology.rho_crit(snap_HR)
        
        #density_plot(snap_HR)
        r, rho = compute_density_profile(snap_HR)
        #plot_density_profile(r, rho)
        r200c[s] = compute_r200c(r, rho, rho_crit)
        #print(r200c)
        print(s)
    np.savetxt("Halo004_r200c.txt", r200c)
    #part = symlib.Particles(sim_dir, include=["E_sph"]) ## for nonMW there is just E. E_sph/E in subhalo ref frame. E>0 unbound.
    """
    h, hist = symlib.read_subhalos(sim_dir)
    #print(len(h[0]))
    h, _ = symlib.read_subhalos(sim_dir)
    a = symlib.scale_factors(sim_dir)
    print(a)

    Mvir = h[0,:]
    Rvir = h[0,:]
    Mvir_z = np.zeros(244)
    Rvir_z = np.zeros(244)
    for i in range(244):
        #print(Mvir[i][1], Rvir[i][7])
        Mvir_z[i] = Mvir[i][1]
        Rvir_z[i] = Rvir[i][7]

    param = symlib.simulation_parameters(sim_dir) # or part.params
    #print(param.keys())
    #mass_dm = param["mp"]/param["h100"] # Msun physical
    #print("dm_mass:", mass_dm)
    #np.savetxt("halo_004_Mvir_Rvir.txt", np.array([Mvir_z, Rvir_z]).T)
    
    #SNAP = 235
    
    edges = np.logspace(0.1, 2.5, 101)
    density_profile = np.zeros((244,100))
    density_profile_smooth = np.zeros((244,100))
    for SNAP in range(243, 244):
        p = part.read(SNAP)
        print(p[0])
        fig = density_projections(p[0], R=Rvir, lim=300, vmax=10000)
        plt.savefig("test_density.png", bbox_inches='tight')
        plt.close()
                # # particles ids associated with the LMC throughout history/current step.
        particle_id_LMC          = p[1]['id']
        particle_id_bound_struct = np.hstack([return_bound_particle_ids(p, subhalo_index) for subhalo_index in trange(2, len(p))])
        to_remove_id             = np.unique(np.hstack([particle_id_LMC, particle_id_bound_struct]))

        # # particles ids that are associated with the main branch
        host_keep_idx = indices_to_keep(p[0]['id'], to_remove_id)

        # pos, vel = p[0]["x"][is_smooth], p[0]["v"][is_smooth] ## 0 is the main host in each snapshot. 
        pos = p[0]["x"]## 0 is the main host in each snapshot. 
        is_smooth = p[0]["smooth"] ## particles that are accreted smoothly 
        pos_smooth = p[0]["x"][is_smooth]
        # Compute density profile
        profile = nba.structure.Profiles(pos, edges)
        profile_smooth = nba.structure.Profiles(pos_smooth, edges)
        rbins, density_profile[SNAP] = profile.density(smooth=1, mass=mass_dm)
        _, density_profile_smooth[SNAP] = profile_smooth.density(smooth=1, mass=mass_dm)
        """
    #np.savetxt("halo_004_density_profile.txt", density_profile)
    #np.savetxt("halo_004_density_profile_smooth.txt", density_profile_smooth)
    # Make image
    
