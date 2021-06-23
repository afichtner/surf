"""
Compute sensitivity kernels for Rayleigh waves.

:copyright:
    Andreas Fichtner (andreas.fichtner@erdw.ethz.ch), December 2020
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""

def kernels_psv(r, r1, r2, r3, r4, _omega, _k, U, I1, rho, A, C, F, L, N, write_output, output_directory, tag):
	"""
	Compute and write sensitivity kernels for Rayleigh waves.

	kernels_psv(r, r1, r2, r3, r4, _omega, _k, U, I1, rho, A, C, F, L, N, write_output, output_directory, tag)
	"""

	import numpy as np

	#- initialisations ----------------------------------------------------------------------------

	K_rho_0 = np.zeros(len(r))
	K_A_0 = np.zeros(len(r))
	K_C_0 = np.zeros(len(r))
	K_F_0 = np.zeros(len(r))
	K_L_0 = np.zeros(len(r))
	K_N_0 = np.zeros(len(r))

	K_rho = np.zeros(len(r))
	K_vph = np.zeros(len(r))
	K_vpv = np.zeros(len(r))
	K_vsh = np.zeros(len(r))
	K_vsv = np.zeros(len(r))
	K_eta = np.zeros(len(r))

	#- phase velocity

	c = _omega / _k

	#- velocities and eta

	vpv = np.sqrt(C/rho)
	vph = np.sqrt(A/rho)
	vsh = np.sqrt(N/rho)
	vsv = np.sqrt(L/rho)
	eta = F / (A-2*L)

	#- compute fundamental kernels ----------------------------------------------------------------

	if (I1!=0.0):

		dr1 = (r2 + _k * F * r3)/C
		cU = c/U

		K_rho_0 = -((rho * cU)/(2.0 * I1)) * (r1**2 + r3**2)
		K_A_0 = (A * r3**2 * cU)/(2.0 * c**2 * I1)
		K_C_0 = (C * cU * dr1**2)/(2.0 * _omega**2 * I1)
		K_F_0 = -(_k * F * cU * dr1 * r3)/(_omega**2 * I1)
		K_L_0 = (cU * r4**2)/(2.0 * _omega**2 * I1 * L)
		K_N_0 = np.zeros(len(r))

	#- compute relative kernels in velocity parametrisation ---------------------------------------

	K_vph = 2*K_A_0 + 2*A*K_F_0/(A-2.0*L)
	K_vpv = 2*K_C_0
	K_vsh = 2*K_N_0
	K_vsv = 2*K_L_0 - 4*L*K_F_0/(A-2.0*L)
	K_rho = K_rho_0 + K_A_0 + K_C_0 + K_N_0 + K_L_0 + K_F_0 
	K_eta = K_F_0

	#- write results to file ----------------------------------------------------------------------

	if write_output:

		f=_omega/(2.0*np.pi)
		cc=_omega/_k

		identifier = f"f={f:.3f}"+f".c={cc:.3f}"
		fid = open(output_directory+"kernels_psv."+tag+"."+identifier,"w")
		fid.write("number of vertical sampling points\n")
		fid.write(str(len(r))+"\n")
		fid.write("radius K_rho_0 K_A_0 K_C_0 K_F_0 K_L_0 K_N_0 K_rho K_vph K_vpv K_vsv K_vsh K_eta\n")
		for idx in np.arange(len(r)):
			fid.write(str(r[idx])+" "+str(K_rho_0[idx])+" "+str(K_A_0[idx])+" "+str(K_C_0[idx])+" "+str(K_F_0[idx])+" "+str(K_L_0[idx])+" "+str(K_N_0[idx])+" "+str(K_rho[idx])+" "+str(K_vph[idx])+" "+str(K_vpv[idx])+" "+str(K_vsh[idx])+" "+str(K_vsv[idx])+" "+str(K_eta[idx])+"\n")
		fid.close()