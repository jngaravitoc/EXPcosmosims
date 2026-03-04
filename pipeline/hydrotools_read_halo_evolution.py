import os
import glob
import sys
import h5py
import numpy as np
from hydrotools.core import interface as iface_run
from hydrotools.common import simulations as common_sims
from hydrotools.common import fields as common_fields


def read_particle_data(h5file):
    """
    Read particle data from an HDF5 file, including optional fuzz and OSHS
    components in either coordinates or velocities.

    Parameters
    ----------
    h5file : h5py.File

    Returns
    -------
    data : dict
        Keys may include:
        - pos, vel
        - fuzz_pos, fuzz_vel
        - oshs_pos, oshs_vel
    """

    data = {
        "pos": [],
        "vel": [],
        "fuzz_pos": [],
        "fuzz_vel": [],
        "oshs_pos": [],
        "oshs_vel": [],
    }

    for key in h5file.keys():
        arr = np.array(h5file[key])

        if key.endswith("_Coordinates"):
            if "fuzz" in key:
                data["fuzz_pos"].append(arr)
            elif "oshs" in key:
                data["oshs_pos"].append(arr)
            else:
                data["pos"].append(arr)

        elif key.endswith("_Velocities"):
            if "fuzz" in key:
                data["fuzz_vel"].append(arr)
            elif "oshs" in key:
                data["oshs_vel"].append(arr)
            else:
                data["vel"].append(arr)

    # Stack where present, else None
    for k in data:
        if len(data[k]) > 0:
            data[k] = np.vstack(data[k])
        else:
            data[k] = None

    return data


def write_one_single_hdf5(
    filename, 
    snapshot,
    redshift,
    time,
    pos,
    vel,
    fuzz_pos=None,
    fuzz_vel=None,
    oshs_pos=None,
    oshs_vel=None,
):
    """
    Write particle data to a single HDF5 file with metadata.

    Metadata is written as file-level attributes.
    """

    with h5py.File(filename, "w") as f:

        # -------------------------
        # Metadata
        # -------------------------
        f.attrs["snapshot"] = snapshot
        f.attrs["redshift"] = redshift
        f.attrs["time"] = time

        # -------------------------
        # Main particles
        # -------------------------
        f.create_dataset("Coordinates", data=pos)
        f.create_dataset("Velocities", data=vel)

        # -------------------------
        # Optional components
        # -------------------------
        if fuzz_pos is not None:
            f.create_dataset("Fuzz_Coordinates", data=fuzz_pos)

        if fuzz_vel is not None:
            f.create_dataset("Fuzz_Velocities", data=fuzz_vel)

        if oshs_pos is not None:
            f.create_dataset("OSHS_Coordinates", data=oshs_pos)

        if oshs_vel is not None:
            f.create_dataset("OSHS_Velocities", data=oshs_vel)

def write_complementary_fields(
    filename,
    m200,
    r200,
    *,
    snapshot,
    redshift,
    time,
):
    """
    Write halo-level complementary fields to HDF5.
    """

    with h5py.File(filename, "w") as f:
        f.attrs["snapshot"] = snapshot
        f.attrs["redshift"] = redshift
        f.attrs["time"] = time

        f.create_dataset("Group_M_Crit200", data=m200)
        f.create_dataset("Group_R_Crit200", data=r200)



def find_particle_keys(h5file):
    """
    Find particle coordinate and velocity datasets in an HDF5 file.

    Parameters
    ----------
    h5file : h5py.File

    Returns
    -------
    coord_keys : list of str
        Dataset names ending in '_Coordinates'.
    vel_keys : list of str
        Dataset names ending in '_Velocities'.
    """

    coord_keys = []
    vel_keys = []

    found_fuzz = False
    found_oshs = False

    for key in h5file.keys():
        if key.endswith("_Coordinates"):
            coord_keys.append(key)
        elif key.endswith("_Velocities"):
            vel_keys.append(key)

        if "fuzz" in key:
            found_fuzz = True
        if "oshs" in key:
            found_oshs = True

    if found_fuzz:
        print("✓ Fuzz particles found")

    if found_oshs:
        print("✓ OSHS particles found")

    return coord_keys, vel_keys


if __name__ == "__main__":

    ###############################################################################
    # Input parameters
    ###############################################################################

    #DATAFILE = './galaxies_tng50-3-dark_099.hdf5'
    SIMULATION = "tng35-3-dark"
    #SNAP_IDX = 99

    IDS_DATAFILE = f'./galaxies_{SIMULATION}_099_ids.hdf5'

    #UTPATH_DIR = "./processed_halos"
    filename = "test_halo.hdf5"
    #s.makedirs(OUTPATH_DIR, exist_ok=True)

    # TODO: update for tng35-dark if needed
    PARTICLE_MASS = 3.2e8
    SUBFIND_IDS_FILE = "tng35-3-dark_halo_sample.txt"

    
    data = np.loadtxt(SUBFIND_IDS_FILE)
    
    snaps = data[:,0]
    time = data[:,2]
    redshift = data[:,1]
    subfind_ids = data[:,4]

    all_coords = []
    all_vels = []

    all_fuzz_coords = []
    all_fuzz_vels = []

    all_oshs_coords = []
    all_oshs_vels = []

    for i in range(90, 99):
    #    try:
        print(i, int(subfind_ids[i]))
        iface_run.extractGalaxyData(
            num_processes = 12, 
            machine_name='umdastro',
            sim = SIMULATION, 
            snap_idx = i,
            no_snapshots = False, 
            paranoid = True, 
            verbose = False,
            mass_selection_type='idxs',
            sh_idxs=[int(subfind_ids[i])],
            output_path = None, 
            buffered_output = False, 
            output_compression = 'gzip',
            extract_satellites = True,
            n_max_extract=None,
            catsh_get = True, 
            catsh_fields = common_fields.default_catsh_fields_dark, 
            catgrp_get = True, 
            catgrp_fields = common_fields.default_catgrp_fields_all,
            tree_get=True,
            tree_fields=['subfind_id', 'is_primary'],
            ptldm_get = True, 
            ptldm_fields = ['Coordinates', 'Velocities','Masses','oshs_Coordinates', 'fuzz_Coordinates'], 
            ptl_in_rad_get = False, 
            #ave_ptl_sep = True, 
            #ptl_rad = None, # 1.5 
            ptl_rad_units = '200m',
            profile_get = False, 
            profile_fields = [])
     #except (Exception):
     #continue
   
        """
        snapshot_file = f"galaxies_tng50-3-dark_{i:03d}.hdf5"

        with h5py.File(snapshot_file, "r") as f:
            coord_keys, vel_keys = find_particle_keys(f)
            coords = [np.array(f[k]) for k in coord_keys]
            vels   = [np.array(f[k]) for k in vel_keys]

            
            for k in coord_keys:
                arr = np.array(f[k])
                if "fuzz" in k:
                    all_fuzz_coords.append(arr)
                elif "oshs" in k:
                    all_oshs_coords.append(arr)
                else:
                    all_coords.append(arr)
    
            for k in vel_keys:
                arr = np.array(f[k])
                if "fuzz" in k:
                    all_fuzz_vels.append(arr)
                elif "oshs" in k:
                    all_oshs_vels.append(arr)
                else:
                    all_vels.append(arr)
         
    write_one_single_hdf5(
        filename, snaps, redshift, time,
        all_coords, all_vels,  
        all_fuzz_coords, all_fuzz_vels,
        all_oshs_coords, all_oshs_vels)
    """
