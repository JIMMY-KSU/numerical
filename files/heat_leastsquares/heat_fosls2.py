"""
Heat FOSLS^2 Coefficient Generator
This script models the Galerkin Weak Formulation of the First-Order System (FOSLS) 
equations for the Heat Equation. The necessary u and V values are imported from the 
original FOSLS approximation for the Heat Equation. It uses the FEniCS project (primarily DOLFIN)
to generate functions representing phi_u and phi_v, the trial function
representations for u and V in the original FOSLS equation. These functions are computed for
each time step using the FOSLS approximation technique. 
It then determines the norm of the approximation and outputs it to the calling function.
The size of the mesh, the time step, and the number of time steps can be 
specified by the calling function.

Molly Q. Feldman
Tufts University/Swarthmore College
9 August 2013
"""
#import necessary libraries used herein
import numpy as np
from dolfin import *
import scipy
from scipy.sparse import *

def main(n, tintervals, dt, uks, vks):
	#defines the mesh for the FEM 
	mesh = UnitSquareMesh(n,n)
	
	#define the Function Spaces on which phi_u and phi_V are defined
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
	
	#defines the trial and test functions for the FOSLS model
	#q represents phi_u and W represents phi_V
	q, W = TrialFunctions(C)
	r, S = TestFunctions(C)
	
	#defines the a equation and assembles its representative matrix
	a = inner(((1/dt)*q - div(W)), ((1/dt)*r - div(S)))*dx + inner((grad(q) - W), \
	 (grad(r) - S))*dx + inner(curl(W), curl(S))*dx
	A = assemble(a)
	
	#sets up the solver, in this case a Krylov indirect solver
	solver = KrylovSolver("gmres", "ml_amg")
	solver.set_operators(A, A)
	
	#variables used in the time loop
	bb = None 
	norms = []
	for i in range(1, len(uks)):
		#f and k represent the evaluated FEniCS functions for u and V at each time step
		f = uks[i]
		k = vks[i]
		
		#defines L, the equation which generates the right-hand side of system, and 
		#generates its associated matrix (the tensor=bb option allows for memory 
		#reuse, providing a small speed advantage)
		L = inner((f + (1/dt)*s1), ((1/dt)*r - div(S)))*dx + inner(k, grad(r) - S)*dx
		bb = assemble(L, tensor = bb)

		#initializes the solution vector
		bigU = Function(C)
	
		#applies the necessary boundary conditions to the system and then solves the 
		#system for bigU	
		[bc.apply(A,bb) for bc in v_bound]
		solver.solve(bigU.vector(), bb)
	
		#computes the norm of the system and then adds it to a list containing norms
		#for each time step
		norm = np.linalg.norm(bb.array() - np.dot(A.array(), bigU.vector().array()))
		norms.append(norm)
	
	return norms
	
	
	
	
	
	
	
	
	
	
	
	