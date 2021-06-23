"""
Computation dispersion curves, stress-displacement functions
and sensitivity kernels for SH propagation.

:copyright:
    Andreas Fichtner (andreas.fichtner@erdw.ethz.ch), December 2020
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import TOOLS.read_xml as rxml
import TOOLS.integrate_sh as ish
import TOOLS.group_velocity_sh as vg
import TOOLS.kernels_sh as ksh
import MODELS.models as m


def dispersion_sh(xml_input):
	"""
	Compute dispersion curves, displacement functions and kernels for SH propagation.
	periods, phase_velocities = dispersion_sh(xml_input)
	Dispersion curves and displacement functions are written to files.
	"""

	#- read input ---------------------------------------------------------------------------------

	inp = rxml.read_xml(xml_input)
	inp = inp[1]

	verbose = int(inp["verbose"])

	model = inp["model"]

	write_output = inp["output"]["write_output"]
	output_directory = inp["output"]["directory"]
	tag = inp["output"]["tag"]

	r_min = float(inp["integration"]["starting_radius"])
	dr = float(inp["integration"]["radius_sampling"])

	f_min = float(inp["f_c_sampling"]["f_min"])
	f_max = float(inp["f_c_sampling"]["f_max"])
	df = float(inp["f_c_sampling"]["df"])

	c_min = float(inp["f_c_sampling"]["c_min"])
	c_max = float(inp["f_c_sampling"]["c_max"])
	dc = float(inp["f_c_sampling"]["dc"])

	#- initialisations ----------------------------------------------------------------------------

	f = np.arange(f_min,f_max + df,df,dtype=float)
	c = np.arange(c_min,c_max + dc,dc,dtype=float)
	omega = 2.0 * np.pi * f

	mode = []
	frequencies = []
	phase_velocities = []
	group_velocities = []

	r = np.arange(r_min, 6371000.0 + dr, dr, dtype=float)
	rho = np.zeros(len(r))
	A = np.zeros(len(r))
	C = np.zeros(len(r))
	F = np.zeros(len(r))
	L = np.zeros(len(r))
	N = np.zeros(len(r))

	for n in np.arange(len(r)):
		rho[n], A[n], C[n], F[n], L[n], N[n] = m.models(r[n], model)

	#- root-finding algorithm ---------------------------------------------------------------------

	#- loop over frequencies
	for _f in f:

		_omega = 2.0 * np.pi * _f
		k = _omega / c
		mode_count = 0.0

		#- loop over trial wavenumbers
		l_left = 0.0
		for n in np.arange(len(k)):

			#- compute vertical wave functions
			l1, l2, r = ish.integrate_sh(r_min, dr, _omega, k[n], model)
			l_right = l2[len(l2)-1]

			#- check if there is a zero -----------------------------------------------------------
			if l_left * l_right < 0.0:

				mode_count += 1.0
				mode.append(mode_count)

				#- start bisection algorithm
				ll_left = l_left
				ll_right = l_right
				k_left = k[n-1]
				k_right = k[n]

				for i in np.arange(5):
					k_new = (k_left * np.abs(ll_right) + k_right * np.abs(ll_left)) / (np.abs(ll_left) + np.abs(ll_right))
					l1, l2, r = ish.integrate_sh(r_min, dr, _omega, k_new, model)
					ll = l2[len(l2)-1]
					if ll * ll_left < 0.0:
						k_right = k_new
						ll_right = ll
					elif ll * ll_right < 0.0:
						k_left = k_new
						ll_left = ll
					else:
						continue

				#==================================================================================
				#- compute final vertical wave functions and corresponding velocities and kernels -
				#==================================================================================

				#- stress and displacement functions 

				l1, l2, r = ish.integrate_sh(r_min, dr, _omega, k_new, model)

				#- phase velocity

				frequencies.append(_f)
				phase_velocities.append(_omega/k_new)

				#- group velocity

				U, I1, I3 = vg.group_velocity_sh(l1, l2, r, _omega/k_new, rho, N)
				group_velocities.append(U)

				#- compute kernels and write them to a file

				ksh.kernels_sh(r, l1, l2, _omega, k_new, U, I1, I3, rho, A, C, F, L, N, write_output, output_directory, tag)

				#==================================================================================
				#- screen output and displacement function files ----------------------------------
				#==================================================================================

				cc=_omega/k_new

				#- plot and print to screen
				if verbose:
					message=f"f={_f:.3f}"+f" Hz, c={cc:.3f}"+f" m/s, U={U:.3f}"+" m/s"
					print(message)

				#- write output
				if write_output:
					identifier = f"f={_f:.3f}"+f".c={cc:.3f}"
					fid = open(output_directory+"displacement_sh."+tag+"."+identifier,"w")
					fid.write("number of vertical sampling points\n")
					fid.write(str(len(r))+"\n")
					fid.write("radius displacement stress\n")
					for idx in np.arange(len(r)):
						fid.write(str(r[idx])+" "+str(l1[idx])+" "+str(l2[idx])+"\n")
					fid.close()

			l_left =l_right

	#==============================================================================================
	#- output -------------------------------------------------------------------------------------
	#==============================================================================================

	if write_output:

		#- write dispersion curve
		fid = open(output_directory+"dispersion_sh."+tag,"w")
		for k in np.arange(len(frequencies)):
			fid.write(str(frequencies[k])+" "+str(phase_velocities[k])+" "+str(group_velocities[k])+"\n")
		fid.close()

	#- return -------------------------------------------------------------------------------------

	return frequencies, phase_velocities, group_velocities
