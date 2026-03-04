import numpy as np
import h5py
from hydrotools.core import interface as iface_run
from hydrotools.common import simulations as common_sims
from hydrotools.common import fields as common_fields



if __name__ == "__main__":
    SUITE = 'tng35-3-dark'
    SUITE = 'tng75-3'
    Mmin = 1e12
    Mmax = 3e12
    DATAFILE = './galaxies_tng100-3-dark_099.hdf5'
    iface_run.extractGalaxyData(num_processes = 1, machine_name='umdastro',
                        sim = SUITE , snap_idx = 99, 
                        no_snapshots = False, paranoid = True, verbose = False,
                        mass_selection_type = 'and',
                        output_path = None, buffered_output = False, output_compression = 'gzip',
                        Mdm_min = Mmin, Mdm_max = Mmax, extract_satellites = False,
                        randomize_order = True, rank_by = 'random', n_max_extract = 1,
                        catsh_get = True, catsh_fields = common_fields.default_catsh_fields_dark, 
                        catgrp_get = True, catgrp_fields = common_fields.default_catgrp_fields_all,
                        tree_get = True, tree_fields = ['subfind_id', 'is_primary', 'SubhaloID'],
                        ptldm_get = True, ptldm_fields = ['Coordinates', 'Velocities', 'oshs_Coordinates', 'fuzz_Coordinates'], 
                        ptl_in_rad_get = True, save_ptl_sep = True, ptl_rad = 1.5, ptl_rad_units = '200m',
                        profile_get = False, profile_fields = [])

    f = h5py.File(DATAFILE, 'r')
    print(f.keys())
    subfind_ids = np.array(f['tree_subfind_id'])
    print(np.shape(subfind_ids))
    sh_idxs = np.array(f["catsh_id"])
    subfind_ids = np.array(f['tree_subfind_id'])
    tree_sid  = np.array(f['tree_SubhaloID'])
    snaps = np.array(f['info']['tree_snaps'])
    times = np.array(f['info']['tree_t'])
    redshifts = np.array(f['info']['tree_z'])
    header = (f"A MW-like halo in {SUITE} \n"
               f"mass range {Mmin}-{Mmax} \n"
              "snaps times redshifts subfind_ids")
   

    print(len(subfind_ids))
    nfields = len(subfind_ids) 
    nsnaps = len(snaps)
    data = np.zeros((nsnaps, nfields+3))
    fmt_data = ("%d", "%.3e", "%.8e")

    data[:,0] = snaps
    data[:,1] = redshifts
    data[:,2] = times

    for i in range(nfields):
        data[:,i+3] = subfind_ids[i]
        fmt_data = fmt_data + ("%d",)
        header = header + " " + str(subfind_ids[i][-1]) + " "

    z0_id = subfind_ids[0][-1]
    np.savetxt(
        f"tng35-3-dark_halo_sample.txt", 
        data, 
        fmt=fmt_data, 
        header=header,
    )
