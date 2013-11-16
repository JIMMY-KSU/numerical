"""
Heat FOSLS Least Sqaures Coefficient Generator
This script models the Galerkin Weak Formulation of the First-Order System (FOSLS) 
equations for the Heat Equation. The necessary u and V values are imported from the 
original FOSLS approximation for the Heat Equation. It uses the FEniCS project (primarily DOLFIN)
to generate functions representing phi_u and phi_v, the trial function
representations for u and V in the original FOSLS equation. These functions are computed for
each time step using the Galerkin approximation technique. 
It then determines the norm of the approximation and outputs it to the calling function.
The size of the mesh, the time step, and the number of time steps can be 
specified by the calling function.

Molly Q. Feldman
Tufts University/Swarthmore College
9 August 2013
"""
#imports necessary libraries and files
import numpy as np
from dolfin import *
import scipy
from pyamg import *
from pyamg.gallery import *
import matplotlib.pyplot as py

def main(n, tintervals, dt, uks, vks):
	#defines the mesh for the FEM approximation
	mesh = UnitSquareMesh(n,n)
	
	#define the Function Spaces on which the phi_u and phi_V 
	#representations are defined
	U = FunctionSpace(mesh, "CG", 1)
	D = VectorFunctionSpace(mesh, "CG", 1)
	C = U * D
	######################################################################
	
	#defines boundary conditions for the mesh 
	#(one condition for each side and one for the entire boundary)
	def always(x, on_boundary):
		return on_boundary
	def top(x, on_boundary):
		return x[1] == 1
	def bottom(x, on_boundary):
		return x[1] == 0
	def right(x, on_boundary):
		return x[0] == 1
	def left(x, on_boundary):
		return x[0] == 0
	
	
	#create Dirichlet boundary condition instances. bc0 applies for the
	#entirety of the function space for phi_u/u and the rest apply for
	#one component or the other of the phi_V/V function space
	zero = Constant(0.0)
	bc0 = DirichletBC(C.sub(0), zero, always)
	bc1 = DirichletBC(C.sub(1).sub(1), zero, right)
	bc2 = DirichletBC(C.sub(1).sub(1), zero, left)
	bc3 = DirichletBC(C.sub(1).sub(0), zero, top)
	bc4 = DirichletBC(C.sub(1).sub(0), zero, bottom)
	
	#an array containing all boundary conditions so they can be applied
	#simultaneously
	v_bound = [bc0, bc1, bc2, bc3, bc4]
	######################################################################
	
	#sets up initial condition expression
	u0 = Expression('100*sin(k*pi*x[0])*sin(l*pi*x[1])', k=1, l=2)
	
	#creates the t-1 instance for the phi_u function (expressed here as q)
	h = Function(C)
	s1, s2 = h.split()
	s1 = interpolate(u0, U)
	
	norms = []
	for i in range(1, len(uks)):
		#f and k represent the evaluated FEniCS functions for u and V at each time step
		f = uks[i]
		k = vks[i]
		
		#defines the trial and test functions for the Galerkin model
		#q represents phi_u and W represents phi_V (q1 is not used, just necessary to
		#define the third test function, z, in the proper Function Space)
		q, W = TrialFunctions(C)
		x, y = TestFunctions(C)
		z, q1 = TestFunctions(C) 
			
		#defines a and j, both components of the main matrix for the system, and L, the
		#equation defining the right-hand side of the system
		a = inner(1/dt*q - div(W), x)*dx + inner(W - grad(q), y)*dx
		j =  inner(curl(W), z)*dx
		L = inner(1/dt*s1 + f, x)*dx + inner(k, y)*dx
		
		#assembles all the matrices for the given time step
		A = assemble(a)
		J = assemble(j)
		bb = assemble(L)
		
		#applies boundary conditions to the system (applied
		#uniquely to A and J and to bb twice in order to 
		#incorporate all the necessary restrictions)
		[bc.apply(A,bb) for bc in v_bound]	  
		[bc.apply(J, bb) for bc in v_bound]
			
		#zeroing out the columns removes some duplication in
		#the application of the boundary conditions to both
		#A and J (a result of having to generate the main
		#matrix in multiple parts)
		r = bb.copy()				
		[bc.zero_columns(J, r) for bc in v_bound]
		
		#augments bb to reflect the 0 right hand side of the J
		#matrix (determined by the original equations)
		bb = bb.array()
		addition = np.zeros(bb.shape)
		bb = np.hstack((bb, addition))
		
		#converts A and J to NumPy arrays 
		A = A.array()
		J = J.array() 
		
		#vertically stacks A and J to create the full matrix
		W = np.vstack((A, J))
		
		#solves the given overdetermined linear system
		res = np.linalg.lstsq(W, bb, rcond=-1)
		res = res[0]
	
		#takes the resultant NumPy vector and assigns a FEniCS
		#function its coefficient values. Necessary in order to
		#generate the t-1 function instance for q/phi_u
		bigU = Function(C)
		bigU.vector()[:] = res[:]
		
		#updates the t-1 time step for q
		q, WW = bigU.split(deepcopy=True)
		s1.assign(q)
		
		#calculates the norm of the answer and adds it to a 
		#running list of norms at each time step
		norm = np.linalg.norm(bb - np.dot(W,res))
		norms.append(norm)
			
	return norms
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	