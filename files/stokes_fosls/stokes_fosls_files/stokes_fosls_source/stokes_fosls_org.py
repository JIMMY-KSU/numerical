"""
Stokes FOSLS Coefficient Generator 

(ORIGINAL BOUNDARY CONDITIONS << not dependent on starting condition)

This script uses the FEniCS project (primarily the DOLFIN functionality) in order
to generate both the mass matrix M and the coefficient array u for the system.
The right-hand side vector b is also generated based on the specified initial condition, u0.
The size of the mesh, the time step, and the number of time steps can be 
specified by the calling function.

Molly Q. Feldman
Tufts University/Swarthmore College
8 August 2013
"""
from dolfin import *
import numpy as np


def main(n, tintervals, dt):
	mesh = UnitSquareMesh(n,n)
	
	S1 = FunctionSpace(mesh, "CG", 2)
	
	S2 = FunctionSpace(mesh, "CG", 2)
	S3 = FunctionSpace(mesh, "CG", 2)
	
	S4 = FunctionSpace(mesh, "CG", 2)
	S5 = FunctionSpace(mesh, "CG", 2)
	S6 = FunctionSpace(mesh, "CG", 2)
	
	P = FunctionSpace(mesh, "CG", 2)
	
	S =  MixedFunctionSpace([S1, S2, S3, S4, S5, S6])
	
	#####################################################
	#general mass matrix calculation
	m1 = TrialFunction(S2)
	n1 = TestFunction(S2)
	
	m = inner(m1, n1)*dx 
	M = assemble(m)
	m_array = M.array()
	
	#####################################################
	
	#####################################################
	ss1 = TrialFunction(P)
	ss2 = TrialFunction(P)
	ss3 = TrialFunction(P)
	
	tt1 = TestFunction(P)
	tt2 = TestFunction(P)
	tt3 = TestFunction(P)
	
	st = 2*inner(ss1, tt1)*dx + inner(ss2, tt2)*dx + inner(ss3, tt3)*dx
	T = assemble(st)
	t_array = T.array()
	#####################################################
	
	###############################################################
	def top_bottom(x, on_boundary):
		#return on_boundary
		return x[1] == 1 or x[1] == 0 
	
	# No-slip boundary condition for velocity
	noslip = Constant(0.0)
	
	bc0 = DirichletBC(S.sub(2), noslip, top_bottom)
	bc2 = DirichletBC(S.sub(4), noslip, top_bottom)
	

	def left_right(x, on_boundary):
		 return x[0] == 1 or x[0] == 0
		 
	bc1 = DirichletBC(S.sub(1), noslip, left_right)
	bc3 = DirichletBC(S.sub(5), noslip, left_right)
	
	
	bound = [bc0, bc1, bc2, bc3, bc4, bc5]
	##############################################################
	
	p, u1, u2, v11, v12, v21 = TrialFunctions(S)
	q, w1, w2, z11,z12, z21 = TestFunctions(S)
	
	u00 = Expression('l/k*sin(k*pi*x[0])*cos(l*pi*x[1])', k=1, l=3)
	u01 = Expression('-cos(k*pi*x[0])*sin(l*pi*x[1])', k=1, l=3)
	
	
	#u00 = Expression('l/k*sin(k*pi*x[0])*cos(l*pi*x[1])', l=1, k=3)
	#u01 = Expression('-cos(k*pi*x[0])*sin(l*pi*x[1])', k=1, l=3)
	
	#uu1, update1, uu2, uu3, uu4, uu5 = Function(S)
	#uu6, uu7, update2, uu8, uu9, uu10 = Function(S)
	
	update = Function(S2*S3)
	update1, update2 = update.split()
	
	update1 = interpolate(u00, S2)
	update2 = interpolate(u01, S3)
	
	f = Constant(0.0)

	a =inner((1/dt * u1 + p.dx(0) - v11.dx(0)- v21.dx(1)), (1/dt * w1 + q.dx(0) - z11.dx(0) - z21.dx(1)))*dx + \
		inner((1/dt * u2 + p.dx(1) - v12.dx(0) + v11.dx(1)), (1/dt * w2 + q.dx(1) - z12.dx(0) + z11.dx(1)))*dx + \
		inner((u1.dx(0) + u2.dx(1)), (w1.dx(0) + w2.dx(1)))*dx + \
		inner((v21.dx(0) - v11.dx(1)), (z21.dx(0) - z11.dx(1)))*dx + inner(-v11.dx(0) - v12.dx(1), -z11.dx(0) - z12.dx(1))*dx +\
		inner((u1.dx(0) - v11), (w1.dx(0) - z11))*dx + inner((u2.dx(0) - v12), (w2.dx(0) - z12))*dx + \
		inner((u1.dx(1) - v21), (w1.dx(1) - z21))*dx + inner((u2.dx(1) + v11), (w2.dx(1) + z11))*dx 
		

	#do I need to interpolate here for the test functions as well? i.e. do I need to interpolate on w1/w2 from u00 etc. etc. etc.
	L = inner(((1/dt * update1) + f), (1/dt*w1 + q.dx(0) - z11.dx(0) - z21.dx(1)))*dx + \
		inner(((1/dt * update2) + f), (1/dt*w2 + q.dx(1) - z12.dx(0) + z11.dx(1)))*dx 
	
	#A = assemble(a)
	
	#solver = KrylovSolver(entry, pre)
	#solver.set_operators(A, A)
	
	bigU = Function(S)
	t = 0
	uks = []
	ps = []
	vks = []
	bb= None
	
	while t <= tintervals:
		
		solve(a == L, bigU, bcs=bound,
      solver_parameters={"linear_solver": "lu"})
		
		p, u1, u2, v11, v12, v21 = bigU.split(deepcopy = True)
		update1.assign(u1)
		update2.assign(u2)
		
		uks.append([u1.vector().array(), u2.vector().array()])
		t += 1
		
	return uks, m_array
