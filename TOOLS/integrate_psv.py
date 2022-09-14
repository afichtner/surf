"""
Integrate the Rayleigh wave equations for a specific circular frequency and wavenumber.
Original system, not actually used to solve the eigenvalue problem, but to compute the final stress and displacement functions.

:copyright:
    Andreas Fichtner (andreas.fichtner@erdw.ethz.ch), September 2022
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""

#--------------------------------------------------------------------------------------------------
#- right-hand sides of the first-order system for Rayleigh waves
#--------------------------------------------------------------------------------------------------

def f1(C,F,k,r2,r3):
	return (r2 / C + k * F * r3 / C)

def f2(rho,omega,k,r1,r4):
	return (-rho * omega**2 * r1 + k * r4)

def f3(L,k,r1,r4):
	return (r4 / L - k * r1)

def f4(rho,A,C,F,omega,k,r2,r3):
	return (-k * F * r2 / C + (k**2 * (A - F**2 / C) - rho * omega**2) * r3)

#--------------------------------------------------------------------------------------------------
#- numerical integration
#--------------------------------------------------------------------------------------------------

def integrate_psv(r_min, dr, omega, k, model, initial_condition):
	"""
	Integrate first-order system for a fixed circular frequency omega and a fixed wavenumber k.
	r1, r2, r3, r4, r = integrate_psv(r_min, dr, omega, k, model)

	r_min:		minimum radius in m
	dr:			radius increment in m
	omega:		circular frequency in Hz
	k:			wave number in 1/m
	model:		Earth model, e.g. "PREM", "GUTENBERG", ... .

	r1, ...:	variables of the Rayleigh wave system
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

	r1 = np.zeros(len(r))
	r2 = np.zeros(len(r))
	r3 = np.zeros(len(r))
	r4 = np.zeros(len(r))

	#- check if phase velocity is below S velocity ------------------------------------------------
	if (1==1): #(k**2 - (omega**2 * rho / L)) > 0.0:

		#- set initial values
		if initial_condition == 1:
			r1[0] = 0.0	
			r2[0] = 1.0 
			r3[0] = 0.0
			r4[0] = 0.0
		else:
			r1[0] = 0.0	
			r2[0] = 0.0 
			r3[0] = 0.0
			r4[0] = 1.0

		#- integrate upwards with 4th-order Runge-Kutta--------------------------------------------

		for n in np.arange(len(r)-1):

			#- compute Runge-Kutta coeficients for l1 and l2

			K1_1 = f1(C[n],F[n],k,r2[n],r3[n])
			K2_1 = f2(rho[n],omega,k,r1[n],r4[n])
			K3_1 = f3(L[n],k,r1[n],r4[n])
			K4_1 = f4(rho[n],A[n],C[n],F[n],omega,k,r2[n],r3[n])

			K1_2 = f1(C[n],F[n],k,r2[n]+0.5*K2_1*dr,r3[n]+0.5*K3_1*dr)
			K2_2 = f2(rho[n],omega,k,r1[n]+0.5*K1_1*dr,r4[n]+0.5*K4_1*dr)
			K3_2 = f3(L[n],k,r1[n]+0.5*K1_1*dr,r4[n]+0.5*K4_1*dr)
			K4_2 = f4(rho[n],A[n],C[n],F[n],omega,k,r2[n]+0.5*K2_1*dr,r3[n]+0.5*K3_1*dr)

			K1_3 = f1(C[n],F[n],k,r2[n]+0.5*K2_2*dr,r3[n]+0.5*K3_2*dr)
			K2_3 = f2(rho[n],omega,k,r1[n]+0.5*K1_2*dr,r4[n]+0.5*K4_2*dr)
			K3_3 = f3(L[n],k,r1[n]+0.5*K1_2*dr,r4[n]+0.5*K4_2*dr)
			K4_3 = f4(rho[n],A[n],C[n],F[n],omega,k,r2[n]+0.5*K2_2*dr,r3[n]+0.5*K3_2*dr)
			
			K1_4 = f1(C[n+1],F[n+1],k,r2[n]+K2_3*dr,r3[n]+K3_3*dr)
			K2_4 = f2(rho[n+1],omega,k,r1[n]+K1_3*dr,r4[n]+K4_3*dr)
			K3_4 = f3(L[n+1],k,r1[n]+K1_3*dr,r4[n]+K4_3*dr)
			K4_4 = f4(rho[n+1],A[n+1],C[n+1],F[n+1],omega,k,r2[n]+K2_3*dr,r3[n]+K3_3*dr)

			#- update

			r1[n + 1] = r1[n] + dr * (K1_1 + 2 * K1_2 + 2 * K1_3 + K1_4) / 6.0
			r2[n + 1] = r2[n] + dr * (K2_1 + 2 * K2_2 + 2 * K2_3 + K2_4) / 6.0
			r3[n + 1] = r3[n] + dr * (K3_1 + 2 * K3_2 + 2 * K3_3 + K3_4) / 6.0
			r4[n + 1] = r4[n] + dr * (K4_1 + 2 * K4_2 + 2 * K4_3 + K4_4) / 6.0

			#- rescale to maximum to prevent overflow

			if initial_condition == 1:
				mm = np.max(np.abs(r2))
			else:
				mm = np.max(np.abs(r4))

			r1 = r1 / mm
			r2 = r2 / mm
			r3 = r3 / mm
			r4 = r4 / mm

	#- return -------------------------------------------------------------------------------------

	return r1, r2, r3, r4, r