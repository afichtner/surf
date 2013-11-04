
import numpy as np
import matplotlib.pyplot as plt

def plot_displacement_sh(filename,show=True):

	"""
		Plot displacement function for Love waves. 
		plot_displacement_sh(filename,show=True)
		filename: filename including path
		show: set to True for showing the plot, and to False for not showing it. show=False can be useful when plotting multiple displacement functions.
	"""

	#- open file and read header information ------------------------------------------------------

	f=open(filename, 'r')

	f.readline()
	n=int(f.readline())
	f.readline()

	l=np.zeros(n)
	r=np.zeros(n)

	#- march through the depth levels and read file -----------------------------------------------

	for k in range(n):
		dummy=f.readline().strip().split(' ')
		r[k]=dummy[0]
		l[k]=dummy[1]

	#- normalise and clean up ---------------------------------------------------------------------

	l=l/np.max(l)
	r=r/1000.0

	f.close()

	#- plot results -------------------------------------------------------------------------------

	plt.plot(l,r,'k')
	plt.xlabel('normalised displacement')
	plt.ylabel('radius [km]')
	
	if show==True: plt.show()