#l2Norm Approximation Method
#Feldman

import heat_galerkin
import scipy
from scipy import integrate
import numpy as np

def main():			
			results = []
			n = 32
			dt = 1/512.0
			tintervals = 52
			
			uks, M = heat_galerkin.main(n, tintervals, dt)
			res = []
			
			for t in range(0, tintervals+1):
				d = uks[t].reshape(n**2, 1)
				val = 0.5 * np.dot(d.T, M*d)
				results.append(val[0][0])
			
			for r in range(0, len(results)):
				print str(dt*(r+1)) + '\t' + str(results[r])
			
			#for r in range(1, len(results)-1):
			#	approx = (results[r] - results[r-1])/(dt*r - dt*(r-1))
			#	print str(dt*r) + '\t' + str(approx)			
main()