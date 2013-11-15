"""
Heat Equation Galerkin Model

This script uses the FEniCS project (primarily the DOLFIN functionality) in order
to generate both the mass matrix M, the coefficient array for u, and the coefficient arrays
for the components of V = laplacian(u) for the system. The right-hand side vector bb is 
also generated based on the specified initial condition, u0.
The size of the mesh, the time step, and the number of time steps can be 
specified by the calling function.
It then computes a user-chosen quantity (energy, relative error etc.) and prints the
results out onto the screen.

Molly Q. Feldman
Tufts University/Swarthmore College
9 August 2013
"""
from dolfin import *
import numpy as np
import math
import matplotlib.pyplot as py
import sys

################################################################################

def sqrt_l2norm(quant, M):
	
	if type(quant) != np.ndarray: 
		quant = quant.vector().array()
	else:
		quant = quant
	
	quant = quant.reshape((quant.shape[0]), 1)
	return math.sqrt(np.dot(quant.T, np.dot(M, quant)))

################################################################################

def l2norm(quant, M):
	
	if type(quant) != np.ndarray: 
		quant = quant.vector().array()
	else:
		quant = quant
	
	quant = quant.reshape((quant.shape[0]), 1)
	return np.dot(quant.T, np.dot(M, quant))[0][0]

################################################################################

def coeff(n, tintervals, dt):
	#define the mesh
	mesh = UnitSquareMesh(n,n)
	
	#define the Function Spaces on which we will be conducting the tests
	U = FunctionSpace(mesh, "CG", 1)
	U1 = FunctionSpace(mesh, "CG", 1)
	
	#################################
	#mass matrix generation
	u = TrialFunction(U)
	v = TestFunction(U)
	m = inner(u, v)*dx
	M = assemble(m)
	
	#stiffness matrix generation
	s = inner(grad(u), grad(v))*dx
	S = assemble(s)
	
	####################################
	#defining boundary conditions
	def always(x, on_boundary):
		return on_boundary
		
	zero = Constant(0.0)
	#boundary conditions for U (zero on all boundaries)
	bc0 = DirichletBC(U, zero, always)
	
	u0 = Expression('100*sin(k*pi*x[0])*sin(l*pi*x[1])', k=1, l=2)
	#generate the u^(k-1) function (which we call w1)
	s1 = Function(U)
	s1 = interpolate(u0, U)
	
	
	u1 = Expression('100*sin(k*pi*x[0])*sin(l*pi*x[1])*pow(E, (-pow(pi, 2)*5*t))', k=1, l=2, E = math.e, t = 0)
	j = Function(U)
	j = interpolate(u1, U)
	
	#set f value (0 in this case)
	f = Constant(0.0)
	
	#sets up equations for a and L; mathematical expressions of the FOSLS equations
	a = inner((1/dt)*u, v)*dx + inner(grad(u), grad(v))*dx
	L = inner((1/dt)*s1, v)*dx

	#FEniCS assembly of the matrix related to a
	A = assemble(a)

	#sets up the FEniCS Krylov solver with options for the type of solver and the 
	#type of preconditioner
	solver = KrylovSolver("cg", "petsc_amg")
	solver.set_operators(A, A)

	#necessary initialization for variables used in the time loop
	bigU = Function(U)
	tt = 1
	uks = []
	js = []
	bb = None
	
	uks.append(s1.vector().array())
	js.append(j.vector().array())
	
	M = M.array()
	S = S.array()
	
	#solves for u and V for each time step
	while tt <= tintervals:
		#using the tensor = bb option allows for better memory management
		bb = assemble(L, tensor = bb)
		
		#applies the boundary condition for each time step to the entire system
		#then solve for bigU 
		bc0.apply(A,bb) 
		solver.solve(bigU.vector(), bb)
		
		u_nodal = bigU.vector()
		u_array= u_nodal.array()
		uks.append(u_array)
	
		u1.t = dt*tt
		j = interpolate(u1, U)
		js.append(j.vector().array())
		
		new = sqrt_l2norm(bigU, M)
		old = sqrt_l2norm(uks[tt-1], M)
		
		print old, new, new/old, 1/(1+ dt*5*(pi**2))
		
		s1.assign(bigU)
		tt += 1
		
	return uks, js, M, S 
	
def main():
	print 
	n = int(sys.argv[1])
	dt = 1/float(sys.argv[2])
	tintervals = int(sys.argv[3])
	
	uks, js, M, S = coeff(n, tintervals, dt)
	
	for i in range(0, tintervals+1):
		
		#r = js[i] - uks[i]
	
		#generates the error 
		#val = l2norm(r, M)
		
		#val = -l2norm(uks[i], S)
		
		#generates the energy
		val = 0.5 * l2norm(uks[i], M)
		
		#generates the relative error
		#val  = sqrt_l2norm(r, M)/sqrt_l2norm(js[i], M)
		
		print str(dt*i) + '\t' + str(val)
		
main()
	

	
	
	
	