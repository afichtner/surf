"""
Compute group velocity for Rayleigh waves.

:copyright:
    Andreas Fichtner (andreas.fichtner@erdw.ethz.ch), November 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""

def group_velocity_psv(r1, r2, r3, r4, r, k, phase_velocity, rho, A, C, F, L, N):
	"""
	Compute Rayleigh wave group velocity for a given set of eigenfunctions (l1, l2), and
	phase velocity (phase_velocity). 

	U, I1, I3 = group_velocity_psv(r1, r2, r3, r4, r, phase_velocity, rho, A, C, F, L, N)
	"""

	import numpy as np

	#- compute the integrals I1 and I3 ------------------------------------------------------------

	I1 = 0.0
	I3 = 0.0
	dr = r[1] - r[0]

	for n in np.arange(len(r)-1):

		I1 = I1 + rho[n] * (r1[n]**2 + r3[n]**2) 
		I3 = I3 + ((A[n]-F[n]**2/C[n])*r3[n]**2 + (r1[n]*r4[n]-F[n]*r2[n]*r3[n]/C[n])/k) 

	I1 = dr * I1
	I3 = dr * I3	

	#- return values ------------------------------------------------------------------------------

	U = I3 / (phase_velocity * I1)

	return U, I1, I3