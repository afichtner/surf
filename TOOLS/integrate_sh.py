"""
Integrate the Love wave equations.

:copyright:
    Andreas Fichtner (andreas.fichtner@erdw.ethz.ch), September 2022
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

	#- Packages.
	import numpy as np
	import MODELS.models as m
	import matplotlib.pyplot as plt

	#- Radius of the Earth (or other planetary body) [m].
	Re = 6371000.0

	#- initialisation -----------------------------------------------------------------------------

	if model == "EXTERNAL":
		r, rho, A, C, F, L, N = m.models(None, model)

	else:
		r = np.arange(r_min, Re + dr, dr, dtype=float)
		rho = np.zeros(len(r))
		A = np.zeros(len(r))
		C = np.zeros(len(r))
		F = np.zeros(len(r))
		L = np.zeros(len(r))
		N = np.zeros(len(r))
		for n in np.arange(len(r)):
			rho[n], A[n], C[n], F[n], L[n], N[n] = m.models(r[n], model)

	l1 = np.zeros(len(r))
	l2 = np.zeros(len(r))

	#- check if phase velocity is below S velocity ------------------------------------------------
	if (1==1): #(k**2 - (omega**2 * rho / L)) > 0.0:

		#- set initial values
		l2[0] = 1.0 
		l1[0] = 0.0

		#- integrate upwards with 4th-order Runge-Kutta--------------------------------------------

		for n in np.arange(len(r)-1):

			#- compute Runge-Kutta coeficients for l1 and l2

			K1_1 = f1(L[n], l2[n])
			K2_1 = f2(N[n], rho[n], k, omega, l1[n]) 

			K1_2 = f1(L[n], l2[n] + K2_1 * dr / 2.0)
			K2_2 = f2(N[n], rho[n], k, omega, l1[n] + K1_1 * dr / 2.0)

			K1_3 = f1(L[n], l2[n] + K2_2 * dr / 2.0)
			K2_3 = f2(N[n], rho[n], k, omega, l1[n] + K1_2 * dr / 2.0)

			K1_4 = f1(L[n+1], l2[n] + K2_3 * dr)
			K2_4 = f2(N[n+1], rho[n+1], k, omega, l1[n] + K1_3 * dr)

			#- update

			l1[n + 1] = l1[n] + dr * (K1_1 + 2 * K1_2 + 2 * K1_3 + K1_4) / 6.0
			l2[n + 1] = l2[n] + dr * (K2_1 + 2 * K2_2 + 2 * K2_3 + K2_4) / 6.0

			#- rescale to maximum to prevent overflow

			l1 = l1 / np.max(np.abs(l2))
			l2 = l2 / np.max(np.abs(l2))

	#- return -------------------------------------------------------------------------------------

	return l1, l2, r