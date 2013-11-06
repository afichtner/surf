
import numpy as np
import matplotlib.pyplot as plt

def plot_kernels(filename, show=True):

	"""
		Plot sensitivity kernels for Love or Rayleigh waves. 
		plot_kernels_sh(filename,show=True)
		filename: filename including path
		show: set to True for showing the plot, and to False for not showing it. show=False can be useful when plotting multiple displacement functions.
	"""

	#- open file and read header information ------------------------------------------------------

	f=open(filename, 'r')

	f.readline()
	n=int(f.readline())
	f.readline()

	r=np.zeros(n)
	K_rho_0=np.zeros(n)
	K_A_0=np.zeros(n)
	K_C_0=np.zeros(n)
	K_F_0=np.zeros(n)
	K_L_0=np.zeros(n)
	K_N_0=np.zeros(n)

	K_rho=np.zeros(n)
	K_vsh=np.zeros(n)
	K_vsv=np.zeros(n)
	K_vph=np.zeros(n)
	K_vpv=np.zeros(n)
	K_eta=np.zeros(n)

	line=np.zeros(n)
	
	#- march through the depth levels and read file -----------------------------------------------

	for k in range(n):
		dummy=f.readline().strip().split(' ')
		r[k]=dummy[0]
		K_rho_0[k]=dummy[1]
		K_A_0[k]=dummy[2]
		K_C_0[k]=dummy[3]
		K_F_0[k]=dummy[4]
		K_L_0[k]=dummy[5]
		K_N_0[k]=dummy[6]

		K_rho[k]=dummy[7]
		K_vph[k]=dummy[8]
		K_vpv[k]=dummy[9]
		K_vsh[k]=dummy[10]
		K_vsv[k]=dummy[11]
		K_eta[k]=dummy[12]


	#- normalise and clean up ---------------------------------------------------------------------

	r=r/1000.0

	f.close()

	#- plot results -------------------------------------------------------------------------------

	#- extremal values

	rmin=np.min(r)
	rmax=6371.0

	valmax_0=1.1*np.max([np.max(np.abs(K_rho_0)), np.max(np.abs(K_A_0)), np.max(np.abs(K_C_0)), np.max(np.abs(K_F_0)), np.max(np.abs(K_L_0)), np.max(np.abs(K_N_0))])
	valmax=1.1*np.max([np.max(np.abs(K_rho)), np.max(np.abs(K_vph)), np.max(np.abs(K_vpv)), np.max(np.abs(K_vsh)), np.max(np.abs(K_vsv)), np.max(np.abs(K_eta))])

	#- fundamental kernels

	plt.subplot(2,3,1)
	plt.plot(K_rho_0,r,'k')
	K_rho_0[n-1]=0.0
	plt.fill(K_rho_0,r,'k',alpha=0.1)
	plt.plot(line,r,'k--')
	plt.ylim([rmin,rmax])
	plt.xlim([-valmax_0,valmax_0])
	plt.xlabel('density')
	plt.ylabel('radius [km]')
	plt.grid(True)

	plt.subplot(2,3,2)
	plt.plot(K_A_0,r,'k')
	K_A_0[n-1]=0.0
	plt.fill(K_A_0,r,'k',alpha=0.1)
	plt.plot(line,r,'k--')
	plt.ylim([rmin,rmax])
	plt.xlim([-valmax_0,valmax_0])
	plt.xlabel('A')
	plt.ylabel('radius [km]')
	plt.grid(True)

	plt.subplot(2,3,3)
	plt.plot(K_C_0,r,'k')
	K_C_0[n-1]=0.0
	plt.fill(K_C_0,r,'k',alpha=0.1)
	plt.plot(line,r,'k--')
	plt.ylim([rmin,rmax])
	plt.xlim([-valmax_0,valmax_0])
	plt.xlabel('C')
	plt.ylabel('radius [km]')
	plt.grid(True)

	plt.subplot(2,3,4)
	plt.plot(K_F_0,r,'k')
	K_F_0[n-1]=0.0
	plt.fill(K_F_0,r,'k',alpha=0.1)
	plt.plot(line,r,'k--')
	plt.ylim([rmin,rmax])
	plt.xlim([-valmax_0,valmax_0])
	plt.xlabel('F')
	plt.ylabel('radius [km]')
	plt.grid(True)

	plt.subplot(2,3,5)
	plt.plot(K_L_0,r,'k')
	K_L_0[n-1]=0.0
	plt.fill(K_L_0,r,'k',alpha=0.1)
	plt.plot(line,r,'k--')
	plt.ylim([rmin,rmax])
	plt.xlim([-valmax_0,valmax_0])
	plt.xlabel('L')
	plt.ylabel('radius [km]')
	plt.grid(True)

	plt.subplot(2,3,6)
	plt.plot(K_N_0,r,'k')
	K_N_0[n-1]=0.0
	plt.fill(K_N_0,r,'k',alpha=0.1)
	plt.plot(line,r,'k--')
	plt.ylim([rmin,rmax])
	plt.xlim([-valmax_0,valmax_0])
	plt.xlabel('N')
	plt.ylabel('radius [km]')
	plt.grid(True)
	
	if show==True: plt.show()

	#- kernels in velocity parametrisation

	plt.subplot(2,3,1)
	plt.plot(K_rho,r,'k')
	K_rho[n-1]=0.0
	plt.fill(K_rho,r,'k',alpha=0.1)
	plt.plot(line,r,'k--')
	plt.ylim([rmin,rmax])
	plt.xlim([-valmax,valmax])
	plt.xlabel('density')
	plt.ylabel('radius [km]')
	plt.grid(True)

	plt.subplot(2,3,2)
	plt.plot(K_vph,r,'k')
	K_vph[n-1]=0.0
	plt.fill(K_vph,r,'k',alpha=0.1)
	plt.plot(line,r,'k--')
	plt.ylim([rmin,rmax])
	plt.xlim([-valmax,valmax])
	plt.xlabel('vph')
	plt.ylabel('radius [km]')
	plt.grid(True)

	plt.subplot(2,3,3)
	plt.plot(K_vpv,r,'k')
	K_vpv[n-1]=0.0
	plt.fill(K_vpv,r,'k',alpha=0.1)
	plt.plot(line,r,'k--')
	plt.ylim([rmin,rmax])
	plt.xlim([-valmax,valmax])
	plt.xlabel('vpv')
	plt.ylabel('radius [km]')
	plt.grid(True)

	plt.subplot(2,3,4)
	plt.plot(K_vsh,r,'k')
	K_vsh[n-1]=0.0
	plt.fill(K_vsh,r,'k',alpha=0.1)
	plt.plot(line,r,'k--')
	plt.ylim([rmin,rmax])
	plt.xlim([-valmax,valmax])
	plt.xlabel('vsh')
	plt.ylabel('radius [km]')
	plt.grid(True)

	plt.subplot(2,3,5)
	plt.plot(K_vsv,r,'k')
	K_vsv[n-1]=0.0
	plt.fill(K_vsv,r,'k',alpha=0.1)
	plt.plot(line,r,'k--')
	plt.ylim([rmin,rmax])
	plt.xlim([-valmax,valmax])
	plt.xlabel('vsv')
	plt.ylabel('radius [km]')
	plt.grid(True)

	plt.subplot(2,3,6)
	plt.plot(K_eta,r,'k')
	K_eta[n-1]=0.0
	plt.fill(K_eta,r,'k',alpha=0.1)
	plt.plot(line,r,'k--')
	plt.ylim([rmin,rmax])
	plt.xlim([-valmax,valmax])
	plt.xlabel('eta')
	plt.ylabel('radius [km]')
	plt.grid(True)
	
	if show==True: plt.show()