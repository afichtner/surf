"""
Earth models for surface wave calculations.

:copyright:
    Andreas Fichtner (andreas.fichtner@erdw.ethz.ch), December 2020
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""

#--------------------------------------------------------------------------------------------------
#- wrapper around the various Earth models
#--------------------------------------------------------------------------------------------------

def models(r, model):
	"""
	Wrapper that returns elastic parameters for the various Earth models provided below.
	rho, A, C, F, L, N = models(r, model)
	Everything in SI units, i.e. radius in m
	"""

	if model=="PREM":
		return model_prem(r)

	elif model=="PREM_iso":
		return model_prem_iso(r)

	elif model=="ONELAYER":
		return model_onelayer(r)

	elif model=="GUTENBERG":
		return model_gutenberg(r)

#--------------------------------------------------------------------------------------------------
#- onelayer
#--------------------------------------------------------------------------------------------------

def model_onelayer(r):
	"""
	One-layered Earth model for a radius r in m.
	"""

	#- march through the various depth levels -----------------------------------------------------

	if (r > 6361000.0):
		rho = 2.7
		vpv = 5.8
		vph = vpv
		vsv = 2.0
		vsh = vsv
		eta = 1.0

	else:
		rho = 3.1
		vpv = 7.8
		vph = vpv
		vsv = 3.0
		vsh = vsv
		eta = 1.0

	#- convert to elastic parameters --------------------------------------------------------------

	rho = 1000.0 * rho
	vpv = 1000.0 * vpv
	vph = 1000.0 * vph
	vsv = 1000.0 * vsv
	vsh = 1000.0 * vsh

	A = rho * vph**2
	C = rho * vpv**2
	N = rho * vsh**2
	L = rho * vsv**2
	F = eta * (A - 2 * L)

	return rho, A, C, F, L, N

#--------------------------------------------------------------------------------------------------
#- Gutenberg's Earth model, taken from Aki & Richards, page 279
#--------------------------------------------------------------------------------------------------

def model_gutenberg(r):
	"""
	Gutenberg's Earth model for a radius r in m.
	"""

	import numpy as np

	#- initialisations ----------------------------------------------------------------------------

	#- depth intervals
	d = np.array([0.0, 19.0, 38.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0])
	d = 6371000.0 - 1000.0 * d

	#- density
	rho_array = np.array([2.74, 3.00, 3.32, 3.34, 3.35, 3.36, 3.37, 3.38, 3.39, 3.41, 3.53, 3.46, 3.48, 3.50, 3.53, 3.58, 3.62, 3.69, 3.82, 4.01, 4.21, 4.40, 4.56, 4.63])

	#- vp
	vp_array = np.array([6.14, 6.58, 8.20, 8.17, 8.14, 8.10, 8.07, 8.02, 7.93, 7.85, 7.89, 7.98, 8.10, 8.21, 8.38, 8.62, 8.87, 9.15, 9.45, 9.88, 10.30, 10.71, 11.10, 11.35])

	#- vs
	vs_array = np.array([3.55, 3.80, 4.65, 4.62, 4.57, 4.51, 4.46, 4.41, 4.37, 4.35, 4.36, 4.38, 4.42, 4.46, 4.54, 4.68, 4.85, 5.04, 5.21, 5.45, 5.76, 6.03, 6.23, 6.32])

	#- determine vp, vs and rho for the relevant layer --------------------------------------------

	for k in np.arange(len(rho_array)):
		if (r <= d[k]) & (r > d[k+1]):
			rho = rho_array[k]
			vpv = vp_array[k]
			vph = vpv
			vsv = vs_array[k]
			vsh = vsv
			eta = 1.0
			continue
		elif r <= 5371000.0:
			rho = 4.63
			vpv = 11.35
			vph = vpv
			vsv = 6.32
			vsh = vsv
			eta = 1.0

	#- convert to elastic parameters --------------------------------------------------------------

	rho = 1000.0 * rho
	vpv = 1000.0 * vpv
	vph = 1000.0 * vph
	vsv = 1000.0 * vsv
	vsh = 1000.0 * vsh

	A = rho * vph**2
	C = rho * vpv**2
	N = rho * vsh**2
	L = rho * vsv**2
	F = eta * (A - 2 * L)

	return rho, A, C, F, L, N


#--------------------------------------------------------------------------------------------------
#- PREM
#--------------------------------------------------------------------------------------------------

def model_prem(r):
	"""
	Return rho, A, C, F, L, N for PREM (Dziewonski & Anderson, PEPI 1981) for 
	a radius r in m. The reference frequency is 1 Hz. Crust continued into the ocean.
	"""

	#- normalised radius
	x = r / 6371000.0

	#- march through the various depth levels -----------------------------------------------------

	#- upper crust
	if (r >= 6356000.0):
		rho = 2.6
		vpv = 5.8
		vph = vpv
		vsv = 3.2
		vsh = vsv
		eta = 1.0

	#- lower crust
	elif (r >= 6346000.6) & (r < 6356000.0):
		rho = 2.9
		vpv = 6.8
		vph = vpv
		vsv = 3.9
		vsh = vsv
		eta = 1.0

	#- LID
	elif (r >= 6291000.0) & (r < 6346000.6):
		rho = 2.6910 + 0.6924 * x
		vpv = 0.8317 + 7.2180 * x
		vph = 3.5908 + 4.6172 * x
		vsv = 5.8582 - 1.4678 * x
		vsh = -1.0839 + 5.7176 * x
		eta = 3.3687 - 2.4778 * x

	#- LVZ
	elif (r >= 6151000.0) & (r < 6291000.0):
		rho = 2.6910 + 0.6924 * x
		vpv = 0.8317 + 7.2180 * x
		vph = 3.5908 + 4.6172 * x
		vsv = 5.8582 - 1.4678 * x
		vsh = -1.0839 + 5.7176 * x
		eta = 3.3687 - 2.4778 * x

	#- Transition zone 1
	elif (r >= 5971000.0) & (r < 6151000.0):
		rho = 7.1089 - 3.8045 * x
		vpv = 20.3926 - 12.2569 * x
		vph = vpv
		vsv = 8.9496 - 4.4597 * x
		vsh = vsv
		eta = 1.0

	#- Transition zone 2
	elif (r >= 5771000.0) & (r < 5971000.0):
		rho = 11.2494 - 8.0298 * x
		vpv = 39.7027 - 32.6166 * x
		vph = vpv
		vsv = 22.3512 - 18.5856 * x
		vsh = vsv
		eta = 1.0

	#- Transition zone 3
	elif (r >= 5701000.0) & (r < 5771000.0):
		rho = 5.3197 - 1.4836 * x
		vpv = 19.0957 - 9.8672 * x
		vph = vpv
		vsv = 9.9839 - 4.9324 * x
		vsh = vsv
		eta = 1.0

	#- Lower mantle 1
	elif (r >= 5600000.0) & (r < 5701000.0):
		rho = 7.9565 - 6.4761 * x + 5.5283 * x**2 - 3.0807 * x**3
		vpv = 29.2766 - 23.6027 * x + 5.5242 * x**2 - 2.5514 * x**3
		vph = vpv
		vsv = 22.3459 - 17.2473 * x - 2.0834 * x**2 + 0.9783 * x**3
		vsh = vsv
		eta = 1.0 

	#- Lower mantle 2
	elif (r >= 3630000.0) & (r < 5600000.0):
		rho = 7.9565 - 6.4761 * x + 5.5283 * x**2 - 3.0807 * x**3
		vpv = 24.9520 - 40.4673 * x + 51.4832 * x**2 - 26.6419 * x**3
		vph = vpv
		vsv = 11.1671 - 13.7818 * x + 17.4575 * x**2 - 9.2777 * x**3
		vsh = vsv
		eta = 1.0

	#- Lower mantle 3
	elif (r >= 3480000.0) & (r < 3630000.0):
		rho = 7.9565 - 6.4761 * x + 5.5283 * x**2 - 3.0807 * x**3
		vpv = 15.3891 - 5.3181 * x + 5.5242 * x**2 - 2.5514 * x**3
		vph = vpv
		vsv = 6.9254 + 1.4672 * x - 2.0834 * x**2 + 0.9783 * x**3
		vsh = vsv
		eta = 1.0

	#- Outer core
	elif (r >= 1221000.5) & (r < 3480000.0):
		rho = 12.5815 - 1.2638 * x - 3.6426 * x**2 - 5.5281 * x**3
		vpv = 11.0487 - 4.0362 * x + 4.8023 * x**2 - 13.5732 * x**3
		vph = vpv
		vsv = 0.0
		vsh = 0.0
		eta = 1.0

	#- Inner Core
	elif (r >= 0.0) & (r < 1221000.5):
		rho = 13.0885 - 8.8381 * x**2
		vpv = 11.2622 - 6.3640 * x**2
		vph = vpv
		vsv = 3.6678 - 4.4475 * x**2
		vsh = vsv
		eta = 1.0 

	#- convert to elastic parameters --------------------------------------------------------------

	rho = 1000.0 * rho
	vpv = 1000.0 * vpv
	vph = 1000.0 * vph
	vsv = 1000.0 * vsv
	vsh = 1000.0 * vsh

	A = rho * vph**2
	C = rho * vpv**2
	N = rho * vsh**2
	L = rho * vsv**2
	F = eta * (A - 2 * L)

	return rho, A, C, F, L, N

#--------------------------------------------------------------------------------------------------
#- PREM isotropic
#--------------------------------------------------------------------------------------------------

def model_prem_iso(r):
	"""
	Return rho, A, C, F, L, N for PREM isotropic (Dziewonski & Anderson, PEPI 1981) for 
	a radius r in m. The reference frequency is 1 Hz. Crust continued into the ocean.
	"""

	#- normalised radius
	x = r / 6371000.0

	#- march through the various depth levels -----------------------------------------------------

	#- upper crust
	if (r >= 6356000.0):
		rho = 2.6
		vpv = 5.8
		vph = vpv
		vsv = 3.2
		vsh = vsv
		eta = 1.0

	#- lower crust
	elif (r >= 6346000.6) & (r < 6356000.0):
		rho = 2.9
		vpv = 6.8
		vph = vpv
		vsv = 3.9
		vsh = vsv
		eta = 1.0

	#- LID
	elif (r >= 6291000.0) & (r < 6346000.6):
		rho = 2.6910 + 0.6924 * x
		vpv = 4.1875 + 3.9382 * x
		vph = vpv
		vsv = 2.1519 + 2.3481 * x
		vsh = vsv
		eta = 3.3687 - 2.4778 * x

	#- LVZ
	elif (r >= 6151000.0) & (r < 6291000.0):
		rho = 2.6910 + 0.6924 * x
		vpv = 4.1875 + 3.9382 * x
		vph = vpv
		vsv = 2.1519 + 2.3481 * x
		vsh = vsv
		eta = 3.3687 - 2.4778 * x

	#- Transition zone 1
	elif (r >= 5971000.0) & (r < 6151000.0):
		rho = 7.1089 - 3.8045 * x
		vpv = 20.3926 - 12.2569 * x
		vph = vpv
		vsv = 8.9496 - 4.4597 * x
		vsh = vsv
		eta = 1.0

	#- Transition zone 2
	elif (r >= 5771000.0) & (r < 5971000.0):
		rho = 11.2494 - 8.0298 * x
		vpv = 39.7027 - 32.6166 * x
		vph = vpv
		vsv = 22.3512 - 18.5856 * x
		vsh = vsv
		eta = 1.0

	#- Transition zone 3
	elif (r >= 5701000.0) & (r < 5771000.0):
		rho = 5.3197 - 1.4836 * x
		vpv = 19.0957 - 9.8672 * x
		vph = vpv
		vsv = 9.9839 - 4.9324 * x
		vsh = vsv
		eta = 1.0

	#- Lower mantle 1
	elif (r >= 5600000.0) & (r < 5701000.0):
		rho = 7.9565 - 6.4761 * x + 5.5283 * x**2 - 3.0807 * x**3
		vpv = 29.2766 - 23.6027 * x + 5.5242 * x**2 - 2.5514 * x**3
		vph = vpv
		vsv = 22.3459 - 17.2473 * x - 2.0834 * x**2 + 0.9783 * x**3
		vsh = vsv
		eta = 1.0 

	#- Lower mantle 2
	elif (r >= 3630000.0) & (r < 5600000.0):
		rho = 7.9565 - 6.4761 * x + 5.5283 * x**2 - 3.0807 * x**3
		vpv = 24.9520 - 40.4673 * x + 51.4832 * x**2 - 26.6419 * x**3
		vph = vpv
		vsv = 11.1671 - 13.7818 * x + 17.4575 * x**2 - 9.2777 * x**3
		vsh = vsv
		eta = 1.0

	#- Lower mantle 3
	elif (r >= 3480000.0) & (r < 3630000.0):
		rho = 7.9565 - 6.4761 * x + 5.5283 * x**2 - 3.0807 * x**3
		vpv = 15.3891 - 5.3181 * x + 5.5242 * x**2 - 2.5514 * x**3
		vph = vpv
		vsv = 6.9254 + 1.4672 * x - 2.0834 * x**2 + 0.9783 * x**3
		vsh = vsv
		eta = 1.0

	#- Outer core
	elif (r >= 1221000.5) & (r < 3480000.0):
		rho = 12.5815 - 1.2638 * x - 3.6426 * x**2 - 5.5281 * x**3
		vpv = 11.0487 - 4.0362 * x + 4.8023 * x**2 - 13.5732 * x**3
		vph = vpv
		vsv = 0.0
		vsh = 0.0
		eta = 1.0

	#- Inner Core
	elif (r >= 0.0) & (r < 1221000.5):
		rho = 13.0885 - 8.8381 * x**2
		vpv = 11.2622 - 6.3640 * x**2
		vph = vpv
		vsv = 3.6678 - 4.4475 * x**2
		vsh = vsv
		eta = 1.0 

	#- convert to elastic parameters --------------------------------------------------------------

	rho = 1000.0 * rho
	vpv = 1000.0 * vpv
	vph = 1000.0 * vph
	vsv = 1000.0 * vsv
	vsh = 1000.0 * vsh

	A = rho * vph**2
	C = rho * vpv**2
	N = rho * vsh**2
	L = rho * vsv**2
	F = eta * (A - 2 * L)

	return rho, A, C, F, L, N