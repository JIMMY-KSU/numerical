#Stokes Galerkin Generator (calls the necessary function)

#import mestokes
import stokes_galerkin
from scipy.sparse import *
from scipy import integrate
import numpy as np

def main():			
			n = 64
			#time step
			dt = 1/1024.0
			tintervals = 103
						
			#number of individual time steps
			
			#an array containing all the necessary time vectors for the run
			#the number of time steps
			#the problem size
			#uks, A, coor = experiment.main(n, tintervals, dt)
			uks, A = stokes_galerkin.main(n, tintervals, dt)
			
			results = []
			#results1 = []
			#ans = []
			A = csr_matrix(A)
			#print coor
			#full = []
			#square = []
			#print len(uks)
			#print len(coor)
			#for w in range(0, len(coor)):
			#	full.append(coor[w][0])
			#	square.append(coor[w][1])
			
			for t in range(0, tintervals+1):
				d = uks[t][0].reshape((uks[t][0].shape[0]), 1)
				e = uks[t][1].reshape((uks[t][1].shape[0]), 1)	
					
				val = 0.5* np.dot(d.T, A*d)
				val2 = 0.5*np.dot(e.T, A*e)
				print val + val2
				#val3 = np.dot(d.T, S*d)
				#val4 = np.dot(e.T, S*e)
				
				#results1.append(val3[0][0] + val4[0][0])
				results.append(val[0][0] + val2[0][0])
			
			#print len(results)		
			print 
			print "************************************************"
			print "PROBLEM SIZE=",n, "TIME STEP SIZE IS=", dt
			print "************************************************"
			print
			for r in range(1, len(results)-1):
				#approx = (results[r] - results[r-1])/(dt*r - dt*(r-1))
				#ans.append(approx)
				
				print  str(dt*(r+1)) + '\t' + str(results[r])#approx)
				#print "[", dt*r, ", ", approx, "],", 
			#print
			#print
			#print 
			#print
			#for q in range(1, len(results1) - 1):
			#	approx = (results1[q] - results[q-1])/(dt*q - dt*(q-1))
			#	print str(dt*q) + '\t' + str(approx)
			
			#return n, np.array(full), np.array(square), np.array(uks)

main()