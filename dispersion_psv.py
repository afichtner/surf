"""
Computation dispersion curves and stress-displacement functions for PSV propagation.

:copyright:
    Andreas Fichtner (andreas.fichtner@erdw.ethz.ch), August 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""

import numpy as np
import matplotlib.pyplot as plt
import TOOLS.read_xml as rxml
import TOOLS.integrate_psv_alt as ipsv_alt
import TOOLS.integrate_psv as ipsv
import TOOLS.group_velocity_psv as vg
import TOOLS.kernels_psv as kpsv
import MODELS.models as m

def dispersion_psv(xml_input):
	"""
	Compute dispersion curves and displacement functions for PSV propagation.
	periods, phase_velocities = dispersion_psv(xml_input)
	Dispersion curves and displacement functions are written to files.
	"""

	#- read input ---------------------------------------------------------------------------------

	inp = rxml.read_xml(xml_input)
	inp = inp[1]

	verbose = int(inp["verbose"])
	plot_dispersion = int(inp["plot_dispersion"])
	plot_amplitudes = int(inp["plot_amplitudes"])

	model = inp["model"]

	write_output = inp["output"]["write_output"]
	output_directory = inp["output"]["directory"]
	tag = inp["output"]["tag"]

	r_min = float(inp["integration"]["starting_radius"])
	dr = float(inp["integration"]["radius_sampling"])

	T_min = float(inp["T_c_sampling"]["T_min"])
	T_max = float(inp["T_c_sampling"]["T_max"])
	dT = float(inp["T_c_sampling"]["dT"])

	c_min = float(inp["T_c_sampling"]["c_min"])
	c_max = float(inp["T_c_sampling"]["c_max"])
	dc = float(inp["T_c_sampling"]["dc"])

	#- initialisations ----------------------------------------------------------------------------

	T = np.arange(T_min,T_max + dT,dT,dtype=float)
	c = np.arange(c_min,c_max + dc,dc,dtype=float)
	omega = 2 * np.pi / T

	mode = []
	periods = []
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

	#- loop over angular frequencies
	for _omega in omega:
		
		mode_count = 0.0
		k = _omega / c

		#- loop over trial wavenumbers
		r_left = 0.0
		for n in np.arange(len(k)):

			#- compute vertical wave functions using alternative system
			r1, r2, r3, r4, r5, r = ipsv_alt.integrate_psv_alt(r_min, dr, _omega, k[n], model)
			r_right = r2[len(r2)-1]

			#- check if there is a zero -----------------------------------------------------------
			if r_left * r_right < 0.0:

				mode_count += 1.0
				mode.append(mode_count)

				#- start bisection algorithm
				rr_left = r_left
				rr_right = r_right
				k_left = k[n-1]
				k_right = k[n]

				for i in np.arange(5):
					k_new = (k_left * np.abs(rr_right) + k_right * np.abs(rr_left)) / (np.abs(rr_left) + np.abs(rr_right))
					r1, r2, r3, r4, r5, r = ipsv_alt.integrate_psv_alt(r_min, dr, _omega, k_new, model)
					rr = r2[len(r2)-1]
					if rr * rr_left < 0.0:
						k_right = k_new
						rr_right = rr
					elif rr * rr_right < 0.0:
						k_left = k_new
						rr_left = rr
					else:
						continue

				#==================================================================================
				#- compute final vertical wave functions and corresponding velocities and kernels -
				#==================================================================================

				#- compute final vertical wave functions using the original first-order system
				#- two independent solutions
				r11, r21, r31, r41, r = ipsv.integrate_psv(r_min, dr, _omega, k_new, model, 1)
				r12, r22, r32, r42, r = ipsv.integrate_psv(r_min, dr, _omega, k_new, model, 2)
				#- determine their weights via boundary condition (weight q1 is set to 1)
				nr = len(r) - 1
				q2 = -r21[nr] / r22[nr]
				#- final solution with boundary condition
				r1 = r11 + q2 * r12
				r2 = r21 + q2 * r22
				r3 = r31 + q2 * r32
				r4 = r41 + q2 * r42
				#- normalise to vertical displacement at the surface
				mm = r1[nr]
				r1 = r1 / mm
				r2 = r2 / mm
				r3 = r3 / mm
				r4 = r4 / mm

				#- phase velocity

				periods.append(2*np.pi/_omega)
				phase_velocities.append(_omega / k_new)

				#- group velocity

				U, I1, I3 = vg.group_velocity_psv(r1, r2, r3, r4, r, k_new, _omega/k_new, rho, A, C, F, L, N)
				group_velocities.append(U)

				#- kernels

				kpsv.kernels_psv(r, r1, r2, r3, r4, _omega, k_new, I3, rho, A, C, F, L, N, write_output, output_directory, tag)

				#==================================================================================
				#- screen output and displacement function files ----------------------------------
				#==================================================================================

				#- plot and print to screen
				if verbose:
					print "T="+str(2*np.pi/_omega)+" s, c="+str(_omega / k_new)+" m/s, U="+str(U)+" m/s"
				if plot_amplitudes:
					f=plt.figure()
					f.text(0.5,0.95,"T="+str(2*np.pi/_omega)+" s, c="+str(_omega / k_new)+" m/s",horizontalalignment="center",verticalalignment="top")
					plt.subplot(2,2,1)
					plt.plot(r, r1)
					plt.xlabel("radius [m]")
					plt.title("vertical displacement")
					plt.subplot(2,2,2)
					plt.plot(r, r2)
					plt.xlabel("radius [m]")
					plt.title("vertical stress")
					plt.subplot(2,2,3)
					plt.plot(r, r3)
					plt.xlabel("radius [m]")
					plt.title("horizontal displacement")
					plt.subplot(2,2,4)
					plt.plot(r, r4)
					plt.xlabel("radius [m]")
					plt.title("horizontal stress")
					plt.show()

				#- write output
				if write_output:
					identifier = "T="+str(2*np.pi/_omega)+".c="+str(_omega / k_new)
					fid = open(output_directory+"displacement_psv."+tag+"."+identifier,"w")
					fid.write("number of vertical sampling points\n")
					fid.write(str(len(r))+"\n")
					fid.write("radius, vertical displacement, horizontal displacement \n")
					for idx in np.arange(len(r)):
						fid.write(str(r[idx])+" "+str(r1[idx])+" "+str(r3[idx])+"\n")
					fid.close()

			r_left = r_right

	#==============================================================================================
	#- output -------------------------------------------------------------------------------------
	#==============================================================================================

	#- dispersion curve ---------------------------------------------------------------------------

	if write_output:

		#- write dispersion curve
		fid = open(output_directory+"dispersion_psv."+tag,"w")
		for k in np.arange(len(periods)):
			fid.write(str(periods[k])+" "+str(phase_velocities[k])+" "+str(group_velocities[k])+"\n")
		fid.close()

	#- plot ---------------------------------------------------------------------------------------

	if plot_dispersion:

		for n in np.arange(len(periods)):

			plt.plot(periods[n],phase_velocities[n],'ko')
			#if mode[n]==1.0:
			plt.plot(periods[n],group_velocities[n],'ro')
		
		plt.margins(0.2)
		plt.xlabel("period [s]")
		plt.ylabel("phase velocity (black), group velocity (red) [m/s]")
		plt.title("PSV dispersion")
		plt.show()

	#- return -------------------------------------------------------------------------------------

	return periods, phase_velocities
