import sys
import symlib 


user = sys.argv[1]
password = sys.argv[2]

# The base directory where data will be downloaded to.
data_dir = "/mnt/home/nico/ceph/symphony/SymphonyMW/"

# The dataset you want to download.
target = "halos"

# Download the first host halo in the Milky Way-mass suite.
symlib.download_files(user, password, "SymphonyMilkyWay", "Halo023",
        data_dir, target=target)