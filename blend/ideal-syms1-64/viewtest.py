#!/usr/bin/python
from __future__ import print_function # For eprint
from PIL import Image as Im

from os import listdir
from os.path import isdir, isfile, join
import matplotlib.pyplot as plt


def pf(*x, **xx):
	print(*x, **xx)

plotrows = 7
plotcols = 7
whichsubplot=-1
axs=fig=None

def view_img(label, img, show=False):
	global axs
	global fig
	global whichsubplot
	global show_images
	plotrows = 7
	plotcols = 7
	whichsubplot += 1
	if (whichsubplot >= plotrows*plotcols):
		whichsubplot = 0
		pf("Hit enter", end='')
		raw_input("")
	if fig is None:
		fig,axs = plt.subplots(plotrows,plotcols,figsize=(11,11))
		plt.tight_layout()
		plt.ion()   
		plt.pause(0.05) # Calls matplotlib's event loop
		fig.subplots_adjust(hspace=0)
		fig.subplots_adjust(wspace=.3)
	yax = int(whichsubplot/plotcols)
	xax = int(whichsubplot % plotcols)
	fig.delaxes(axs[yax][xax]) 
	newaxs = plt.subplot(plotrows, plotcols, whichsubplot+1)
	axs[yax][xax] = newaxs
	plt.title(label, fontsize=10)
	plt.axis('off')
	axs[yax][xax].imshow(img, cmap="gray", interpolation='nearest') # interpolation='None'
	plt.pause(0.05) # Calls matplotlib's event loop
def load_img(fname):
	img = Im.open(fname)
	return img

for d in listdir("."):
	if isdir(d):
		idf=d + "/0001.png"
		bf="../bent-syms1-64/" + d + "/0001.png"
		if not isfile(idf):
			pf("No ideal file for", d, ": ", idf)
		else:
			if not isfile(bf):
				pf("No bent file for", d, ": ", bf)
			else:
				iimg=load_img(idf)
				dimg=load_img(bf)
				view_img(""+d, iimg)
				view_img("b:"+d, dimg)

