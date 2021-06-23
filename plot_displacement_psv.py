
"""
Plotting Rayleigh-wave displacement functions.

:copyright:
    Andreas Fichtner (andreas.fichtner@erdw.ethz.ch), December 2020
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times"
plt.rcParams.update({'font.size': 30})
plt.rcParams['xtick.major.pad']='12'
plt.rcParams['ytick.major.pad']='12'

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

	fig, (ax1, ax2) = plt.subplots(1,2,sharey='row',figsize=(30,50))

	ax1.plot(r1,r,'k')
	ax2.plot(r2,r,'k')

	ax1.grid()
	ax2.grid()

	ax1.set_title('normalised vertical displacement$',pad=30)
	ax1.set(xlabel='$y_1$',ylabel='z [km]')

	ax2.set_title('normalised horizontal displacement',pad=30)
	ax2.set(xlabel='$y_2$')
	

	plt.savefig('psv.png',format='png')

	if show==True: plt.show()