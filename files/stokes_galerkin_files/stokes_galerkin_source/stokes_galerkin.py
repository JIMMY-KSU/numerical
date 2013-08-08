#ONE LEVEL STOKES GALERKIN

from dolfin import *
import numpy as np


def main(n,tintervals, dt):
# Load mesh
	mesh = UnitSquareMesh(n,n)
	
	# Define function spaces
	V = VectorFunctionSpace(mesh, "CG", 2)
	V1  = FunctionSpace(mesh, "CG", 2)
	Q = FunctionSpace(mesh, "CG", 1)
	W = V * Q
	
	# Boundaries
	u0 = Expression(('l/k*sin(k*pi*x[0])*cos(l*pi*x[1])', '-cos(k*pi*x[0])*sin(l*pi*x[1])'), k=1, l=3)
	
	#u00 = Expression('l/k*sin(k*pi*x[0])*cos(l*pi*x[1])', k=1, l=3)
	#u01 = Expression('-cos(k*pi*x[0])*sin(l*pi*x[1])', k=1, l=3)
	
	def top_bottom(x, on_boundary):
		#return on_boundary
		return x[1] == 1 or x[1] == 0 
	
	# No-slip boundary condition for velocity
	noslip = Expression('0')
	bc0 = DirichletBC(W.sub(0).sub(1), noslip, top_bottom)
	

	def left_right(x, on_boundary):
		 return x[0] == 1 or x[0] == 0
		 
	bc1 = DirichletBC(W.sub(0).sub(0), noslip, left_right)

	
	w = Function(W)
	w1, w2 = w.split()
	w1 = interpolate(u0, V)
	
	# Define variational problem
	(u, p) = TrialFunctions(W)
	(v, q) = TestFunctions(W)
	
	
	f = Constant((0.0, 0.0))
	
	a = (inner(grad(u), grad(v))*dx) + (div(v)*p*dx) - (q*div(u)*dx) + (1/(dt)*inner(u,v)*dx)
	L = -inner(f, v)*dx + 1/(dt)*inner(w1, v)*dx
	
	# Form for use in constructing preconditioner matrix
	#a better preconditioner might be - inner(grad(p), grad(q)) (but start with what we have below)
	#gmres allows us to have mixed eigenvalues
	b = (1/dt* inner(u, v)*dx) + (inner(grad(u), grad(v))*dx) + (p*q*dx) + ((dt)*inner(grad(p), grad(q))*dx)
	
	##################################################
	#general mass matrix calculation
	m1 = TrialFunction(V1)
	m2 = TestFunction(V1)

	m = inner(m1, m2)*dx 
	M = assemble(m)
	m_array = M.array()
	#print "size m_array", m_array.shape
	
	#stiffness matrix calculation, based off of V1 and m1/m2
	s = inner(grad(m1), grad(m2))*dx
	S = assemble(s)
	s_array = S.array()
	##################################################
	
	#A = assemble(a)
	# Assemble preconditioner system
	##P, btmp = assemble_system(b, L, bc0)
	#P = assemble(b)
	##r = assemble(L)
	#bc0.apply(P, r)
	#bc1.apply(P, r)
		
	#solver = KrylovSolver("gmres", "petsc_amg")

	# Associate operator (A) and preconditioner matrix (P)
	#solver.set_operators(A, P)

	U = Function(W)
	t =0
	uks = []
	ps = []
	total = 0
	bb = None
	#for the necessary length of time
	#print mesh
	
	
	while t <= tintervals:
		#bb = assemble(L, tensor=bb)
		#if t == 0:
		#bc0.apply(A,bb)
		#bc1.apply(A,bb)
		#solver.solve(U.vector(), bb)
		solve(a == L, U, bcs=[bc0, bc1],
      solver_parameters={"linear_solver": "lu"})
		u, p = U.split(deepcopy=True)
		#t += 1
		u1, u2 = u.split(deepcopy=True)
		#print u.vector()
		u1_nodal = u1.vector()
		u1_array = u1_nodal.array()
		u2_nodal = u2.vector()
		u2_array = u2_nodal.array()
		
		uks.append([u1_array, u2_array]) #i.e. we are getting the little u and little v components of the big U vector
		w1.assign(u)
		#if t == 10:
		#		plot(u1)
		#	interactive()
		
		t += 1
		#print X
		#plot(u1)
		#plot(u2)
		#interactive()
	return uks, m_array #u.vector().array(), m_array, coor
	
#main(23, 11, 0.01)

