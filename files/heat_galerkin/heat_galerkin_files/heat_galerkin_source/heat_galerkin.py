#Coefficient Matrix Generator for a Q! Poisson problem with meaningful RHS
#Allows for the compilation of boundary conditions as well as time variant
#constant generation
#Feldman (original code by MacLachlan)
# 9 July 2013

# basic math
import math
# use quadrature from scipy
from scipy.integrate import *

# Use numpy and pyamg for matrix stuff
from pyamg import *
from pyamg.gallery import *
import numpy as np
from numpy.linalg import *
from scipy import sparse

# set RHS function
def rhs_func(x,y):
	#return 1+ x + 2*y
	return 100*math.sin(math.pi*x)*math.sin(2*math.pi*y)

# Define bilinear basis function
def bilinear(x,y,xi1,xi,yj1,yj):
	return (((x-xi1)/(xi-xi1))*((y-yj1)/(yj-yj1)))

# RHS vector is computed as integrals of rhs_func * bilinear
def rhs_prod(y,x,xi1,xi,yj1,yj):
	return rhs_func(x,y)*bilinear(x,y,xi1,xi,yj1,yj)

def main(n, tintervals, dt):

	# uniform nxn grid Poisson stencil for the Heat Equation
	#initializes the grid used in each time step coefficient computation
	stencilA = np.array([[-1/3.0,-1/3.0,-1/3.0],[-1/3.0,8/3.0,-1/3.0],[-1/3.0,-1/3.0,-1/3.0]], dtype=float)
	A = gallery.stencil_grid(stencilA, (n,n), dtype=float, format='csr') 
	h = 1/(1.0+n)
	stencilM = np.array([[1/36.0,1/9.0,1/36.0],[1/9.0,4/9.0,1/9.0],[1/36.0,1/9.0,1/36.0]], dtype=float)

	M = gallery.stencil_grid(h**2 * stencilM, (n,n), dtype=float, format='csr') 
	
	B = M+dt*A
	
	# generate interior node locations
	x_nodes = np.linspace(0,1,n+2)
	
	solution = None
	
	# Analytical solution is sin(pi x)sin(2pi y)
	sol2D = np.zeros(n**2,dtype=float).reshape(n,n)
	for i in range(0,n):
		for j in range(0,n):
			sol2D[i,j] = 100*math.sin(math.pi*x_nodes[i+1])*math.sin(math.pi*x_nodes[j+1])/(5*math.pi**2)
	
	# Reshape as 1D array
	solution = sol2D.reshape(n**2,1)
	#print solution
	# RHS computation

	# initialize to zero
	rhs2D = np.zeros(n**2,dtype=float).reshape(n,n)
	
	for i in range(0,n):
		for j in range(0,n):
			rhs2D[i,j] += dblquad(rhs_prod,x_nodes[i],x_nodes[i+1],lambda x:x_nodes[j],lambda x:x_nodes[j+1],args=(x_nodes[i],x_nodes[i+1],x_nodes[j],x_nodes[j+1]))[0]
			rhs2D[i,j] += dblquad(rhs_prod,x_nodes[i+1],x_nodes[i+2],lambda x:x_nodes[j],lambda x:x_nodes[j+1],args=(x_nodes[i+2],x_nodes[i+1],x_nodes[j],x_nodes[j+1]))[0]
			rhs2D[i,j] += dblquad(rhs_prod,x_nodes[i],x_nodes[i+1],lambda x:x_nodes[j+1],lambda x:x_nodes[j+2],args=(x_nodes[i],x_nodes[i+1],x_nodes[j+2],x_nodes[j+1]))[0]
			rhs2D[i,j] += dblquad(rhs_prod,x_nodes[i+1],x_nodes[i+2],lambda x:x_nodes[j+1],lambda x:x_nodes[j+2],args=(x_nodes[i+2],x_nodes[i+1],x_nodes[j+2],x_nodes[j+1]))[0]
	
	# reshape to vector
	rhs = rhs2D.reshape(n**2,1)

	####################################################################
	#************Heat Equation Coefficient/Time Generation**************
		
	#accumulator for the timestep number of coefficient matrices,
	#which we will then run the approximation for
	uks = []

	#set the first element to be equal to the u^(0)th case	
	# pyamg solve
	ml = ruge_stuben_solver(B)
	x = ml.solve(rhs, tol=1e-10)
	uks.append(x)

	
	for k in range(1, tintervals+1):
		f = M*uks[k-1]
		x = ml.solve(f, tol=1e-10)
		uks.append(x)
		
	# check that things match up- shouldn't be exact, but should get
	# better as n gets large
	#print M
	return uks, M