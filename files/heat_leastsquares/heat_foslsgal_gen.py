""""
FOSLS^2 and FOSLS Least Squares for the Heat Equation Generator
This file generates the norm for our FOSLS^2 and FOSLS Least Squares approximations.
It does them in tandem in order to maximize the efficiency of computation (i.e. the
energy is only calculated once and the original FOSLS approximation for the Heat
Equation is only generated once). For each time step, the norm is computed, the 
norm divided by a scaling factor is given to stdout, and the results is usually 
rerouted to a .data file.

Molly Q. Feldman
Tufts University/Swarthmore College
9 August 2013
"""
#imported files as well libraries
import heat_fosls_helper
import heat_foslsgal
import heat_fosls2
from scipy.sparse import *
from scipy import integrate
import numpy as np
import math
import sys

def main():			
			#dimension to determine the mesh size (creates an n by n mesh)
			n = int(sys.argv[1])
			#time step
			dt = 1/float(sys.argv[2])
			#number of time steps to generate
			tintervals = int(sys.argv[3])
			
			#determines the distance between nodes (important to the accuracy of the
			#approximation)
			h = 1/float(n)
			
			#determines the energy computed at each time step for the given parameters
			#(is the energy that is determined for the original system
			us, vs, uks, vks, M, S = heat_fosls_helper.main(n, tintervals, dt)
			#print "after origin helper"
			bb = heat_foslsgal.main(n, tintervals, dt, us, vs)
			#print "after least-squares"
			b = heat_fosls2.main(n, tintervals, dt, us, vs)
			#print "after fosls^2"
			results = []
			results2 = []
			for t in range(0, tintervals):
				#necessary reshaping for the dot product method
				d = uks[t]
			
				#computes the component-wise kinetic energy
				energy = 0.5* np.dot(d.T, M*d)
				
				#computes the norm divided by a scaling factor for each time step
				val = h
				
				results.append(bb[t]/val)
				results2.append(b[t]/val)
			
			#prints the results to the screen, which are usually piped to a different
			#file
			for k in range(0, len(results) -1):
				print str(dt*(k+1)) + '\t' + str(results[k])
			
			print
			for j in range(0, len(results2) - 1):
				print str(dt*(j+1)) + '\t' + str(results2[j])
			
main()