"""
Integrate the Love wave equations.

:copyright:
    Andreas Fichtner (andreas.fichtner@erdw.ethz.ch), August 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""



#--------------------------------------------------------------------------------------------------
#- displacement function
#--------------------------------------------------------------------------------------------------

def f1(L,l2):
	"""
	Right-hand side of the displacement derivative for SH propagation.
	"""
	return l2 / L

#--------------------------------------------------------------------------------------------------
#- stress function
#--------------------------------------------------------------------------------------------------

def f2(N, rho, k, omega, l1):
	"""
	Right-hand side of the stress derivative for SH propagation.
	"""
	return (N * k**2 - rho * omega**2) * l1

#--------------------------------------------------------------------------------------------------
#- numerical integration
#--------------------------------------------------------------------------------------------------

def integrate_sh(r_min, dr, omega, k, model):
	"""
	Integrate first-order system for a fixed circular frequency omega and a fixed wavenumber k.
	l1, l2, r = integrate_sh(r_min, dr, omega, k, model)

	r_min:		minimum radius in m
	dr:			radius increment in m
	omega:		circular frequency in Hz
	k:			wave number in 1/m
	model:		Earth model, e.g. "PREM", "GUTENBERG", ... .

	l1, l2:		variables of the Love wave system
	r:			radius vector in m
	"""

	import numpy as np
	import MODELS.models as m
	import matplotlib.pyplot as plt

	#- initialisation -----------------------------------------------------------------------------

	r = np.arange(r_min, 6371000.0 + dr, dr, dtype=float)
	l1 = np.zeros(len(r))
	l2 = np.zeros(len(r))

	rho, A, C, F, L, N = m.models(r[0], model)

	#- check if phase velocity is below S velocity ------------------------------------------------
	if (k**2 - (omega**2 * rho / L)) > 0.0:

		#- set initial values
		l2[0] = 1.0 
		l1[0] = 0.0 #L * np.sqrt(k**2 - (omega**2 * rho / L))

		#- integrate upwards with 4th-order Runge-Kutta--------------------------------------------

		for n in np.arange(len(r)-1):

			#- compute Runge-Kutta coeficients for l1 and l2

			rho, A, C, F, L, N = m.models(r[n], model)
			K1_1 = f1(L, l2[n])
			K2_1 = f2(N, rho, k, omega, l1[n]) 

			rho, A, C, F, L, N = m.models(r[n] + dr / 2.0, model)
			K1_2 = f1(L, l2[n] + K2_1 * dr / 2.0)
			K2_2 = f2(N, rho, k, omega, l1[n] + K1_1 * dr / 2.0)

			K1_3 = f1(L, l2[n] + K2_2 * dr / 2.0)
			K2_3 = f2(N, rho, k, omega, l1[n] + K1_2 * dr / 2.0)

			rho, A, C, F, L, N = m.models(r[n] + dr, model)
			K1_4 = f1(L, l2[n] + K2_3 * dr)
			K2_4 = f2(N, rho, k, omega, l1[n] + K1_3 * dr)

			#- update

			l1[n + 1] = l1[n] + dr * (K1_1 + 2 * K1_2 + 2 * K1_3 + K1_4) / 6.0
			l2[n + 1] = l2[n] + dr * (K2_1 + 2 * K2_2 + 2 * K2_3 + K2_4) / 6.0

			#- rescale to maximum to prevent overflow

			l1 = l1 / np.max(np.abs(l2))
			l2 = l2 / np.max(np.abs(l2))

	#- return -------------------------------------------------------------------------------------

	return l1, l2, r