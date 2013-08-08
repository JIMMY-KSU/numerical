import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import math

cmap = plt.get_cmap('spectral')
colors = ['b', 'b', 'b', 'g', 'g', 'g', 'm']
combos = ['ob', '^b', 'sb', 'og', '^g', 'sg', '*m']
yaxes = sys.argv[1]
#circles = np.arange(0, 0.1, 10)
plt.xlabel("Time (secs)", fontsize=15)
plt.ylabel(yaxes, fontsize=15)
plots = []
plots1 = []
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=14)
big = []
bigs = []
for i in range(3, len(sys.argv)):
	
	color = colors[i-3]
	combo = combos[i-3]
	data = open(sys.argv[i])
	data_array = []
	x = []
	
	for line in data:
		entry = []

		parts = line.split()
		x.append(float(parts[0]))
		entry.append(math.log(-float(parts[1])))
		#entry.append(float(parts[2]))
		data_array.append(entry[0])
	
	maxx = max(x)
	
	if sys.argv[i] == 'exact_energy.data' or sys.argv[i] == 'derv.data' or sys.argv[i] == 'energy.data' or sys.argv[i] == 'exact13.data': # 'stokes_galerkin_actual.data':
		range1 = np.arange(x[0], maxx, x[2]-x[1])
	
	else:
		range1 = np.arange(x[0], maxx+(x[2]-x[1]), x[2]-x[1])
	print
	print sys.argv[i]
	print x[0]
	print range1
	print len(range1)
	print len(data_array)
	
	#set the plot title
	plt.title(sys.argv[2], fontsize=15)
	
	#sets the plot color scheme (idk if that's what we want as much)
	plt.summer()
	
	x = i%4
	new = []
	news = []
	for k in range(len(range1)-1):
		if x == 0: 
			new.append(range1[k])
			news.append(data_array[k])
		if x == 4:
			x = 0
		else:
			x += 1
	
	plt.grid(True)
	p1, = plt.plot(range1, data_array, color = color, linewidth=2)
	plots.append(p1)
	big.append(new)
	bigs.append(news)

for q in range(len(bigs)):
	p2, = plt.plot(big[q], bigs[q], combos[q], markersize = 7)
	#if x == 0:
	plots1.append((plots[q], p2))
	#plt.legend(loc = 2, fontsize=18)


plt.legend(plots1, ["n = 16, dt = 1/512", "n = 32, dt = 1/512",  "n = 64, dt = 1/512", "n = 16, dt = 1/1024", \
 "n = 32, dt = 1/1024", "n = 64, dt = 1/1024", "numerical solution"], loc=1, fontsize=16)
plt.show()
