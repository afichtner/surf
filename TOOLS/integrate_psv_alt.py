"""
Integrate the alternative version of the Rayleigh wave equations.
See Takeuchi & Saito (1972), page 257.
Only used to find zeroes of the characteristic function in a stable way.

:copyright:
    Andreas Fichtner (andreas.fichtner@erdw.ethz.ch), August 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""

#--------------------------------------------------------------------------------------------------
#- right-hand sides of the alternative first-order system
#--------------------------------------------------------------------------------------------------

def f1(C,L,r4,r5):
	return (r4 / L - r5 / C)

def f2(rho,A,C,F,omega,k,r4,r5):
	return (-omega**2 * rho * r4 + (omega**2 * rho - k**2 * (A - F**2 / C)) * r5)

def f3(C,F,k,r4,r5):
	return (k * r4 + k * F * r5 / C)

def f4(rho,A,C,F,omega,k,r1,r2,r3):
	return ((-omega**2 * rho + k**2 * (A - F**2 / C)) * r1 + r2 / C - 2 * k * F * r3 / C)

def f5(rho,L,omega,k,r1,r2,r3):
	return (omega**2 * rho * r1 - r2 / L - 2 * k * r3)


#--------------------------------------------------------------------------------------------------
#- numerical integration
#--------------------------------------------------------------------------------------------------

def integrate_psv_alt(r_min, dr, omega, k, model):
	"""
	Integrate first-order system for a fixed circular frequency omega and a fixed wavenumber k.
	r1, r2, r3, r4, r5, r = integrate_psv_alt(r_min, dr, omega, k, model)

	r_min:		minimum radius in m
	dr:			radius increment in m
	omega:		circular frequency in Hz
	k:			wave number in 1/m
	model:		Earth model, e.g. "PREM", "GUTENBERG", ... .

	r1, ...:	variables of the alternative Rayleigh wave system
	r:			radius vector in m
	"""

	import numpy as np
	import MODELS.models as m
	import matplotlib.pyplot as plt

	#- initialisation -----------------------------------------------------------------------------

	r = np.arange(r_min, 6371000.0 + dr, dr, dtype=float)
	r1 = np.zeros(len(r))
	r2 = np.zeros(len(r))
	r3 = np.zeros(len(r))
	r4 = np.zeros(len(r))
	r5 = np.zeros(len(r))

	rho, A, C, F, L, N = m.models(r[0], model)

	#- check if phase velocity is below S velocity ------------------------------------------------
	if (k**2 - (omega**2 * rho / L)) > 0.0:

		#- set initial values
		r1[0] = 0.0	
		r2[0] = 1.0 
		r3[0] = 0.0
		r4[0] = 0.0
		r5[0] = 0.0

		#- integrate upwards with 4th-order Runge-Kutta--------------------------------------------

		for n in np.arange(len(r)-1):

			#- compute Runge-Kutta coeficients for l1 and l2

			rho, A, C, F, L, N = m.models(r[n], model)
			K1_1 = f1(C,L,r4[n],r5[n])
			K2_1 = f2(rho,A,C,F,omega,k,r4[n],r5[n])
			K3_1 = f3(C,F,k,r4[n],r5[n])
			K4_1 = f4(rho,A,C,F,omega,k,r1[n],r2[n],r3[n])
			K5_1 = f5(rho,L,omega,k,r1[n],r2[n],r3[n]) 

			rho, A, C, F, L, N = m.models(r[n] + dr / 2.0, model)
			K1_2 = f1(C,L,r4[n]+0.5*K4_1*dr,r5[n]+0.5*K5_1*dr)
			K2_2 = f2(rho,A,C,F,omega,k,r4[n]+0.5*K4_1*dr,r5[n]+0.5*K5_1*dr)
			K3_2 = f3(C,F,k,r4[n]+0.5*K4_1*dr,r5[n]+0.5*K5_1*dr)
			K4_2 = f4(rho,A,C,F,omega,k,r1[n]+0.5*K1_1*dr,r2[n]+0.5*K2_1*dr,r3[n]+0.5*K3_1*dr)
			K5_2 = f5(rho,L,omega,k,r1[n]+0.5*K1_1*dr,r2[n]+0.5*K2_1*dr,r3[n]+0.5*K3_1*dr) 

			K1_3 = f1(C,L,r4[n]+0.5*K4_2*dr,r5[n]+0.5*K5_2*dr)
			K2_3 = f2(rho,A,C,F,omega,k,r4[n]+0.5*K4_2*dr,r5[n]+0.5*K5_2*dr)
			K3_3 = f3(C,F,k,r4[n]+0.5*K4_2*dr,r5[n]+0.5*K5_2*dr)
			K4_3 = f4(rho,A,C,F,omega,k,r1[n]+0.5*K1_2*dr,r2[n]+0.5*K2_2*dr,r3[n]+0.5*K3_2*dr)
			K5_3 = f5(rho,L,omega,k,r1[n]+0.5*K1_2*dr,r2[n]+0.5*K2_2*dr,r3[n]+0.5*K3_2*dr) 

			rho, A, C, F, L, N = m.models(r[n] + dr, model)
			K1_4 = f1(C,L,r4[n]+K4_3*dr,r5[n]+K5_3*dr)
			K2_4 = f2(rho,A,C,F,omega,k,r4[n]+K4_3*dr,r5[n]+K5_3*dr)
			K3_4 = f3(C,F,k,r4[n]+K4_3*dr,r5[n]+K5_3*dr)
			K4_4 = f4(rho,A,C,F,omega,k,r1[n]+K1_3*dr,r2[n]+K2_3*dr,r3[n]+K3_3*dr)
			K5_4 = f5(rho,L,omega,k,r1[n]+K1_3*dr,r2[n]+K2_3*dr,r3[n]+K3_3*dr) 

			#- update

			r1[n + 1] = r1[n] + dr * (K1_1 + 2 * K1_2 + 2 * K1_3 + K1_4) / 6.0
			r2[n + 1] = r2[n] + dr * (K2_1 + 2 * K2_2 + 2 * K2_3 + K2_4) / 6.0
			r3[n + 1] = r3[n] + dr * (K3_1 + 2 * K3_2 + 2 * K3_3 + K3_4) / 6.0
			r4[n + 1] = r4[n] + dr * (K4_1 + 2 * K4_2 + 2 * K4_3 + K4_4) / 6.0
			r5[n + 1] = r5[n] + dr * (K5_1 + 2 * K5_2 + 2 * K5_3 + K5_4) / 6.0

			#- rescale to maximum to prevent overflow

			mm = np.max(np.abs(r2))
			r1 = r1 / mm
			r2 = r2 / mm
			r3 = r3 / mm
			r4 = r4 / mm
			r5 = r5 / mm

	#- return -------------------------------------------------------------------------------------

	return r1, r2, r3, r4, r5, r