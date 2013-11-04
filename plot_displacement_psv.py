
import numpy as np
import matplotlib.pyplot as plt

def plot_displacement_psv(filename,show=True):

	"""
		Plot displacement functions for Rayleigh waves. 
		plot_displacement_psv(filename,show=True)
		filename: filename including path
		show: set to True for showing the plot, and to False for not showing it. show=False can be useful when plotting multiple displacement functions.
	"""

	#- open file and read header information ------------------------------------------------------

	f=open(filename, 'r')

	f.readline()
	n=int(f.readline())
	f.readline()

	r1=np.zeros(n)
	r2=np.zeros(n)
	r=np.zeros(n)

	#- march through the depth levels and read file -----------------------------------------------

	for k in range(n):
		dummy=f.readline().strip().split(' ')
		r[k]=dummy[0]
		r1[k]=dummy[1]
		r2[k]=dummy[2]

	
	r=r/1000.0

	f.close()

	#- plot results -------------------------------------------------------------------------------

	plt.subplot(1,2,1)
	plt.plot(r1,r,'k')
	plt.xlabel('normalised vertical displacement')
	plt.ylabel('radius [km]')

	plt.subplot(1,2,2)
	plt.plot(r2,r,'k')
	plt.xlabel('normalised horizontal displacement')
	
	if show==True: plt.show()