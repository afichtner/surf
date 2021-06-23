"""
Compute group velocity for Love waves.

:copyright:
    Andreas Fichtner (andreas.fichtner@erdw.ethz.ch), December 2020
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""

def group_velocity_sh(l1, l2, r, phase_velocity, rho, N):
	"""
	Compute Love wave group velocity for a given set of eigenfunctions (l1, l2), and
	phase velocity (phase_velocity). Further input: radius vector (r) and Earth model (rho, N).

	U, I1, I3 = group_velocity_sh(l1, l2, dr, _omega, phase_velocity, model)
	"""

	import numpy as np

	#- compute the integrals I1 and I3 ------------------------------------------------------------

	I1 = 0.0
	I3 = 0.0
	dr = r[1] - r[0]

	for n in np.arange(len(r)):

		I1 = I1 + (rho[n] * l1[n]**2) 
		I3 = I3 + (N[n] * l1[n]**2) 

	I1 = dr * I1
	I3 = dr * I3	

	#- return values ------------------------------------------------------------------------------

	U = I3 / (phase_velocity * I1)

	return U, I1, I3