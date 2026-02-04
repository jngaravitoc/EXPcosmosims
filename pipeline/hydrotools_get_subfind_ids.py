import numpy as np
import h5py
from hydrotools.core import interface as iface_run
from hydrotools.common import simulations as common_sims
from hydrotools.common import fields as common_fields



if __name__ == "__main__":
    SUITE = 'tng35-3-dark'
    Mmin = 1e12
    Mmax = 3e12
    DATAFILE = './galaxies_tng50-3-dark_099.hdf5'
    iface_run.extractGalaxyData(num_processes = 12, machine_name='umdastro',
                        sim = SUITE , snap_idx = 99, 
                        no_snapshots = False, paranoid = True, verbose = False,
                        mass_selection_type = 'and',
                        output_path = None, buffered_output = False, output_compression = 'gzip',
                        Mdm_min = Mmin, Mdm_max = Mmax, extract_satellites = False,
                        randomize_order = True, rank_by = 'random', n_max_extract = 1,
                        catsh_get = True, catsh_fields = common_fields.default_catsh_fields_dark, 
                        catgrp_get = True, catgrp_fields = common_fields.default_catgrp_fields_all,
                        tree_get = True, tree_fields = ['subfind_id', 'is_primary'],
                        ptldm_get = True, ptldm_fields = ['Coordinates', 'Velocities', 'oshs_Coordinates', 'fuzz_Coordinates'], 
                        ptl_in_rad_get = True, save_ptl_sep = True, ptl_rad = 1.5, ptl_rad_units = '200m',
                        profile_get = False, profile_fields = [])


    f = h5py.File(DATAFILE, 'r')
    sh_idxs = f["catsh_id"]
    subfind_ids = np.array(f['tree_subfind_id'])
    snaps = np.array(f['info']['tree_snaps'])
    times = np.array(f['info']['tree_t'])
    redshifts = np.array(f['info']['tree_z'])
    header = (f"A MW-like halo in {SUITE} \n"
               f"mass range {Mmin}-{Mmax} \n"
              "subfund ids, snaps, times, redshifts, sh_idxs")
    print(subfind_ids)
    z0_id = subfind_ids[0][-1]
    np.savetxt(
        f"tng35-3-dark_halo_{z0_id:06d}.txt", 
        np.array([subfind_ids[0], snaps, times, redshifts, sh_idxs[0]]).T, 
        fmt=("%d", "%d", "%.8e", "%.3e", "%d"), 
        header=header,
    )
