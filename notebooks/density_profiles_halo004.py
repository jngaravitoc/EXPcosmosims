import numpy as np
import symlib 
import matplotlib.pyplot as plt
from tqdm import trange
#from matplotlib.colors import SymLogNorm, LogNorm
import nba
#plt.style.use('dark_background')


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

    # SAFE check: only index removed_sorted where idx is in-bounds
    found = np.zeros(host.shape, dtype=bool)
    mask_valid = idx < removed_sorted.size
    if mask_valid.any():
        found[mask_valid] = removed_sorted[idx[mask_valid]] == host[mask_valid]

    keep_mask = ~found
    return np.nonzero(keep_mask)[0]


if __name__ == "__main__":
    SIM_PATH = "/mnt/home/aarora/ceph/Symphony/data/"
    SUITE = 'MWest'
    HALO = "Halo004"
    SNAP = 235
    sim_dir = symlib.get_host_directory(SIM_PATH, SUITE, HALO)                                   
    part = symlib.Particles(sim_dir, include=["E_sph"]) ## for nonMW there is just E. E_sph/E in subhalo ref frame. E>0 unbound.
    p = part.read(SNAP)

    param = symlib.simulation_parameters(sim_dir) # or part.params
    print(param.keys())
    mass_dm = param["mp"]/param["h100"] # Msun physical
    print("dm_mass:", mass_dm)
    
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
    edges = np.logspace(0.1, 2.5, 100)
    profile = nba.structure.Profiles(pos, edges)
    profile_smooth = nba.structure.Profiles(pos_smooth, edges)
    rbins, density_profile = profile.density(smooth=1, mass=mass_dm)
    rbins_smooth, density_profile_smooth = profile_smooth.density(smooth=1, mass=mass_dm)
    np.savetxt("halo_004_density_profile.txt", np.array([rbins, density_profile]).T)
    # Make image
