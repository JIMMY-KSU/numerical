"""
Heat Equation FOSLS Coefficient Generator
This script uses the FEniCS project (primarily the DOLFIN functionality) in order
to generate both the mass matrix M, the coefficient array for u, and the coefficient arrays
for the components of V = laplacian(u) for the system. The right-hand side vector bb is 
also generated based on the specified initial condition, u0.
The size of the mesh, the time step, and the number of time steps can be 
specified by the calling function.

Molly Q. Feldman
Tufts University/Swarthmore College
9 August 2013
"""
from dolfin import *
import numpy as np
import math
import matplotlib.pyplot as py

def main(n, tintervals, dt):
	#define the mesh
	mesh = UnitSquareMesh(n,n)
	
	#define the Function Spaces on which we will be conducting the tests
	U = FunctionSpace(mesh, "CG", 1)
	U1 = FunctionSpace(mesh, "CG", 1)
	D = VectorFunctionSpace(mesh, "CG", 1)
	C = U * D
	
	#################################
	#mass matrix generation
	m1 = TrialFunction(U1)
	m2 = TestFunction(U1)
	m = inner(m1, m2)*dx
	M = assemble(m)
	m_array = M.array()
	
	#stiffness matrix generation
	s = inner(grad(m1), grad(m2))*dx
	S = assemble(s)
	s_array = S.array()
	
	####################################
	#defining boundary conditions
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
		
	zero = Constant(0.0)
	#boundary conditions for U (zero on all boundaries)
	bc0 = DirichletBC(C.sub(0), zero, always)
	
	#boundary conditions for D (i.e. boundary conditions for V)
	#sets boundary conditions according to where each component of V is
	#guaranteed to be zero according to the specifications of the FOSLS equations)
	bc1 = DirichletBC(C.sub(1).sub(1), zero, right)
	bc2 = DirichletBC(C.sub(1).sub(1), zero, left)
	bc3 = DirichletBC(C.sub(1).sub(0), zero, top)
	bc4 = DirichletBC(C.sub(1).sub(0), zero, bottom)
	
	#a list containing all the necessary boundary conditions to apply to the mixed 
	#function space
	v_bound = [bc0, bc1, bc2, bc3, bc4]
	
	#define the test and trial functions
	(u, V) = TrialFunctions(C)
	(q, W) = TestFunctions(C)
	
	
	u0 = Expression('100*sin(k*pi*x[0])*sin(l*pi*x[1])', k=1, l=2)
	
	#generate the u^(k-1) function (which we call w1)
	s = Function(C)
	s1, s2 = s.split()
	s1 = interpolate(u0, U)
	
	#set f value (0 in this case)
	f = Constant(0.0)
	
	#sets up equations for a and L; mathematical expressions of the FOSLS equations
	a = inner(((1/dt)*u - div(V)), ((1/dt)*q - div(W)))*dx + inner((grad(u) - V), (grad(q) - W))*dx + inner(curl(V), curl(W))*dx
	L = inner((f + (1/dt)*s1), ((1/dt)*q - div(W)))*dx
	
	#FEniCS assembly of the matrix related to a
	A = assemble(a)

	#sets up the FEniCS Krylov solver with options for the type of solver and the 
	#type of preconditioner
	solver = KrylovSolver("gmres", "ml_amg")
	solver.set_operators(A, A)

	#necessary initialization for variables used in the time loop
	bigU = Function(C)
	t =0
	us = []
	vs = []
	vks = []
	uks = []
	total = 0
	bb = None
	
	#solves for u and V for each time step
	while t <= tintervals:
		#using the tensor = bb option allows for better memory management
		bb = assemble(L, tensor = bb)
		
		#apply all boundary conditions for each time step to the entire system
		#then solve for bigU (contains solutions for both u and V)
		[bc.apply(A,bb) for bc in v_bound]
		solver.solve(bigU.vector(), bb)
		
		#splits bigU into its individual components and converts the FEniCS
		#objects to their NumPy coefficient arrays for computation
		u, V = bigU.split(deepcopy = True)
		us.append(u)
		vs.append(V)
		uks.append(u.vector().array())
		vks.append(V.vector().array())
		t += 1
		#updates the previous time step for u
		s1.assign(u)
		
	return us, vs, uks, vks, m_array, s_array	