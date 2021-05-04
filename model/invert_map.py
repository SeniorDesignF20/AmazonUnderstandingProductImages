import numpy as np 

def invert_map(heatmap):

	for i in range(heatmap.shape[0]):
		for j in range(heatmap.shape[1]):

			r, g, b = heatmap[i,j]

			temp = b
			b = r
			r = g
			r = temp

			heatmap[i,j] = (r,g,b)

	return heatmap