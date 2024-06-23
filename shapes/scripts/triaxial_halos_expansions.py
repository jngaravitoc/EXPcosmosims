import numpy as np
import pyEXP
import EXPtools 
from EXPtools.basis_builder import makebasis

nparticles = 1000000
halo_name = 'Hernquist_halo_2_1'
HHalo = EXPtools.utils.halo.ICHernquist(size=nparticles)

for theta in range(360):
	if theta == 0:
		add_coef = False
	else: 
		add_coefs = True
	
	sphericalH = HHalo.triaxial(axis_ratios=[2, 1, 1], rot_angle=theta)
	basis, coefs = makebasis(sphericalH, np.ones(nparticles)/nparticles, 
						 basis_model='Hernquist', basis_id='sphereSL', 
						 time=theta, nbins=50, rmin=0.02, rmax=2, log_space=True,
                         lmax=5, nmax=10, scale=1, modelname='SLGrid.empirical_hern_halo',                        cachename=".slgrid_sph_hern_empirical", 
                         add_coef=add_coef, coef_file='hern_2_1_coefs.h5')


#coefs.WriteH5Coefs("coefs_" + halo_name + "_.h5")

