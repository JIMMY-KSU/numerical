"""
Stokes Equation FOSLS Generator
This file calls a stokes coefficient generator
in order to use the dot product formulation of the Newtonian kinetic energy 
equation. It then computes the kinetic energy for each specified time step
and calculates the approximation to its derivative. It finally sends a tab-separated
tuple of the time and the derivative to stdout (typically, results are redirected
into a .data file).

This file has the ability to call two differens stokes coefficient generators:
stokes_fosls_org.py which uses the known boundary conditions from the givens in the
stokes equation and stokes_fosls_new.py which uses both the known boundary conditions
and boundary conditions inherent to the given initial condition.

Molly Feldman
Tufts University/Swarthmore College
13 August 2013
"""

#imports necessary files
import stokes_fosls_org
import stokes_fosls_new
from scipy.sparse import *
from scipy import integrate
import numpy as np
import sys

def main():	
			#creates an n by n mesh for the FEM 	
			n = 16
			#time step
			dt = 1/10.0
			#number of time step calculations to compute
			tintervals = 10
			
			#calls stokes_fosls_org or stokes_fosls_new and generates the
			#necessary coefficient arrays
			uks, M = stokes_fosls_new.main(n, tintervals, dt)
			
			#converts the mass matrix to a NumPy sparse matrix
			M = csr_matrix(M)
			
			#computes the energy for each given time step component-wise
			results = []
			for t in range(0, tintervals+1):
				d = uks[t][0].reshape((uks[t][0].shape[0]), 1)
				e = uks[t][1].reshape((uks[t][1].shape[0]), 1)	
			
				val = 0.5* np.dot(d.T, M*d)
				val2 = 0.5*np.dot(e.T, M*e)
				
				results.append(val[0][0] + val2[0][0])
			
			#############################################################
			#RESULT PRINT OUT
			#uncomment one for loop or the other for either energy value print out
			#or derivative computation 
			
			#PRINT ENERGY
			for r in range(0, len(results)):
				print  str(dt*(r+1)) + '\t' + str(results[r])
				
			#PRINT DERIVATIVE
			for r in range(1, len(results)-1):
				approx = (results[r] - results[r-1])/(dt*r - dt*(r-1))
				print str(dt*r) + '\t' + str(approx)	
				
main()