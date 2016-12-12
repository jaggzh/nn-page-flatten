#!/usr/bin/python
from __future__ import print_function # For eprint
from keras.models import Sequential, Model # , load_weights, save_weights
from keras.layers import Dense, merge, Reshape, UpSampling2D, Flatten, Convolution2D, Deconvolution2D, MaxPooling2D, Input, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.layer_utils import print_summary
from keras.optimizers import Adam, SGD
import numpy as np
import sys
from os import listdir
from os.path import isdir, isfile, join
import matplotlib.pyplot as plt
from random import randint
import random
import re
import os
import time
from time import sleep
from PIL import Image
import matplotlib.image as mpimg
from math import ceil
import math

convact='tanh'

def get_linux_terminal():
	import os
	env = os.environ
	def ioctl_GWINSZ(fd):
		try:
			import fcntl, termios, struct, os
			cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,
		'1234'))
		except:
			return
		return cr
	cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
	if not cr:
		try:
			fd = os.open(os.ctermid(), os.O_RDONLY)
			cr = ioctl_GWINSZ(fd)
			os.close(fd)
		except:
			pass
	if not cr:
		cr = (env.get('LINES', 25), env.get('COLUMNS', 80))

		### Use get(key[, default]) instead of a try/catch
		#try:
		#	cr = (env['LINES'], env['COLUMNS'])
		#except:
		#	cr = (25, 80)
	return int(cr[1]), int(cr[0])

#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

#from keras.datasets import cifar10
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#pf(X_train)
#sys.exit(0)

weight_store_imgdata = "weights-imgdata.h5"
weight_store_angles = "weights-angles.h5"
bent_dir = "blend/pages"
ideal_dir = "blend/ideal"
img_ids_train=[]
img_ids_test=[]
verbose=0
show_images=1
datagen_input=None
datagen_output=None
img_width=67
img_height=67
max_imagesets=1000 # imageset = Each unique page of words (bent or flat) (-1 is unlimited)
#max_main_imgloops = 100 # Number of times to loop through entire set of images
train_epochs=1
out_batch_versions=3 # number of distorted images to feed in
in_batch_versions=3 # number of distorted images to feed in
load_weights=1      # load prior run stored weights
save_weights=1      # load prior run stored weights
test_fraction = .07  # Percentage (well.. fraction) of the data set for the test set
whichsubplot = -1
axs = None
fig = None
normalize = True
#opt_run_test = True
opt_run_test = False

## Functions
def exit(ec):
	sys.exit(ec)
def pf(*x, **y):
	print(*x, **y)
	sys.stdout.flush()
def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)
def vprint(verbosity, *args, **kwargs):
	if (verbose >= verbosity):
		pf(*args, **kwargs)
def load_imgnames():
	vprint(2, "Opening " + bent_dir + "\n")
	i = 0
	for name in listdir(bent_dir):
		i = i+1
		ifile=join(bent_dir, name)
		if isdir(ifile):        # Valid bent_images dir
			                    # ...corresponding to valid ideal_image dir
			if not isdir(join(ideal_dir, name)):
				eprint("Dir " + name + " not in ideal folder: " + ideal_dir)
			if random.random() > test_fraction:
				img_ids_train.append(name)
			else:
				img_ids_test.append(name)
		if max_imagesets > 0 and i > max_imagesets:
			break
def init():
	# fix random seed for reproducibility
	global termwidth, termheight
	termwidth, termheight = get_linux_terminal()
	seed = 9
	random.seed(seed)
	np.random.seed(seed)
	np.set_printoptions(threshold=64, linewidth=termwidth-1, edgeitems=3)
	datagen_input = ImageDataGenerator(
		rotation_range=0,
		width_shift_range=0.09,
		height_shift_range=0.09,
		shear_range=0,
		zoom_range=0.1,
		horizontal_flip=False,
		fill_mode='nearest')
		#featurewise_center=True,
		#featurewise_std_normalization=True)
	datagen_output = ImageDataGenerator(
		rotation_range=0,
		width_shift_range=0.09,
		height_shift_range=0.09,
		shear_range=0,
		zoom_range=0.1,
		horizontal_flip=False,
		fill_mode='nearest')
		#featurewise_center=True,
		#featurewise_std_normalization=True)
	return datagen_input, datagen_output
def imgset_from_dir(dir, id):
	imgset=[]
	imgdir=join(dir, id)
	for f in listdir(imgdir):
		path = join(imgdir, f)
		if isfile(path):
			imgset.append(path)
	return imgset
def imgset_ideal(id):
	return imgset_from_dir(ideal_dir, id)
def imgset_bent(id):
	return imgset_from_dir(bent_dir, id)
def show_shape(inputs, x):
	# we can predict with the model and print the shape of the array.
	dummy_input = np.ones((10, 1, img_width, img_height))

	model = Model(input=inputs, output=x)
	#pf("MODEL SUMMARY:")
	#model.summary()
	#pf("/MODEL SUMMARY:")
	#pf(" MODEL PREDICT: ",)
	preds = model.predict(dummy_input)
	pf(preds.shape)
	#pf(" /MODEL PREDICT:")
def create_nn_test():
	imgwh = img_width*img_height
	act='tanh'
	inputs = Input(shape = (1, img_width, img_height))
	x = Flatten()(inputs); pf("flatten(): "); show_shape(inputs, x)
	dense = Dense(imgwh, activation=act)
	x = dense(x); pf("Dense(256): "); show_shape(inputs, x)
	x = Reshape((1,img_width,img_height))(x); pf("reshape(", (1, img_width, img_height), ") ", sep='', end=''); show_shape(inputs, x)
	outputs = x
	model = Model(input=inputs, output=outputs)
	pf(model.summary())
	pf("final prediction: ", sep='', end=''); show_shape(inputs, outputs)
	pf("Compiling model")
	#sgd=SGD(lr=0.1, momentum=0.000, decay=0.0, nesterov=False)
	opt=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	#dense.trainable = False
	model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
	#model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
	return model

def create_nn2():
	act='tanh'

	inputs = Input(shape = (1, img_width, img_height))
	pf("input(): ", sep='', end=''); show_shape(inputs, inputs)

	x = Flatten()(inputs)
	pf("flatten(): ", sep='', end=''); show_shape(inputs, x)

	angles = Dense(16, activation=act, name="angles")
	pf("dense(16)::: ", sep='', end='');
	imgdata = Dense(1024, activation=act, name="imgdata")
	pf("dense(1024)::: ", sep='', end='');

	angles_called = angles(x)
	imgdata_called = imgdata(x)

	viewer_angles_layer = Reshape((1, 4, 4))(angles_called) # Output for viewing
	viewer_img_layer = Reshape((1, 32, 32))(imgdata_called)    # Output for viewing

	joined = merge([angles_called, imgdata_called], mode='concat', concat_axis=1)

	x = Dense(4489, activation=act)(joined)
	pf("dense(4489): ", sep='', end=''); show_shape(inputs, x)

	x = Reshape((1,img_height,img_width))(x)
	pf("reshape(", (1, img_width, img_height), "): ", sep='', end=''); show_shape(inputs, x)

	outputs=x
	model_train_angles = Model(input=inputs, output=outputs)
	model = Model(input=inputs, output=outputs)
	model_view_angles = Model(input=inputs, output=viewer_angles_layer)
	model_view_imgdata = Model(input=inputs, output=viewer_img_layer)
	pf(model.summary())
	pf("final prediction: ", sep='', end=''); show_shape(inputs, x)

	pf("Compiling models")
	#sgd=SGD(lr=0.1, momentum=0.000, decay=0.0, nesterov=False)

	opt=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	angles.trainable = False
	imgdata.trainable = True
	model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
	angles.trainable = True
	imgdata.trainable = False
	model_train_angles.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
	model_view_angles.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
	model_view_imgdata.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
	pf("Loading weights")
	if load_weights and isfile(weight_store_imgdata):
		model.load_weights(weight_store_imgdata)
	if load_weights and isfile(weight_store_angles):
		model_train_angles.load_weights(weight_store_angles)
	return model, model_train_angles, model_view_angles, model_view_imgdata

def create_nn():
	global activation
	filters = 36

	inputs = Input(shape = (1, img_width, img_height))
	pf("input(): ", sep='', end=''); show_shape(inputs, inputs)

	x = Convolution2D(filters, 2, 2, activation=convact, border_mode='same', subsample=(1,1))(inputs)
	pf("conv2d(", filters, ",2,2): ", sep='', end=''); show_shape(inputs, x)

	x = MaxPooling2D((2,2), border_mode='same', dim_ordering='th')(x)
	pf("maxpool((2,2)): ", sep='', end=''); show_shape(inputs, x)

	filters = 16


	x = Convolution2D(filters, 2, 2, activation=convact, border_mode='same', subsample=(1,1))(x)
	pf("conv2d(", filters, ",2,2): ", sep='', end=''); show_shape(inputs, x)

	x = MaxPooling2D((2,2), border_mode='same', dim_ordering='th')(x)
	pf("maxpool((2,2)): ", sep='', end=''); show_shape(inputs, x)


#	x = Convolution2D(filters, 2, 2, activation=convact, border_mode='same', subsample=(1,1))(x)
#	pf("conv2d(", filters, ",2,2): ", sep='', end=''); show_shape(inputs, x)
#
#	x = MaxPooling2D((2,2), border_mode='same', dim_ordering='th')(x)
#	pf("maxpool((2,2)): ", sep='', end=''); show_shape(inputs, x)


	x = Flatten()(x) # -> 1296
	pf("flatten(): ", sep='', end=''); show_shape(inputs, x)


	x = Dense(96)(x)
	pf("dense(96): ", sep='', end=''); show_shape(inputs, x)

	x = Dense(96)(x)
	pf("dense(96): ", sep='', end=''); show_shape(inputs, x)

#	x = Dense(filters*16)(x)
#	pf("dense(9*filters): ", sep='', end=''); show_shape(inputs, x)
#
#	x = Reshape((filters,4,4))(x)
#	pf("reshape((", 1, ",4,4): ", sep='', end=''); show_shape(inputs, x)

	x = Dense(filters*64)(x)
	pf("dense(filters*64): ", sep='', end=''); show_shape(inputs, x)
	x = Reshape((filters,8,8))(x)
	pf("reshape((", 1, ",8,8): ", sep='', end=''); show_shape(inputs, x)

#	x = UpSampling2D(size = (2,2), dim_ordering='th')(x)
#	pf("upsamp2d((2,2): ", sep='', end=''); show_shape(inputs, x)
#	x = Deconvolution2D(filters,2,2, border_mode='same', subsample=(1,1), output_shape=(None,filters,7,7))(x)
#	pf("deconv2d(", filters, ",2,2): ", sep='', end=''); show_shape(inputs, x)
#	x = ZeroPadding2D(padding=(0, 1, 0, 1), dim_ordering='default')(x)
#	pf("ZeroPadding2D(0,1,0,1): ", sep='', end=''); show_shape(inputs, x)

	x = UpSampling2D(size = (2,2), dim_ordering='th')(x)
	pf("upsamp2d((2,2): ", sep='', end=''); show_shape(inputs, x)
	x = Deconvolution2D(filters,2,2, border_mode='same', subsample=(1,1), output_shape=(None,filters,15,15))(x)
	pf("deconv2d(", filters, ",2,2): ", sep='', end=''); show_shape(inputs, x)
	x = ZeroPadding2D(padding=(1, 0, 1, 0), dim_ordering='default')(x)
	
	filters = 16

	x = UpSampling2D(size = (2,2), dim_ordering='th')(x)
	pf("upsamp2d((2,2): ", sep='', end=''); show_shape(inputs, x)
	x = Deconvolution2D(filters,2,2, border_mode='same', subsample=(1,1), output_shape=(None,filters,31,31))(x)
	pf("deconv2d(", filters, ",2,2): ", sep='', end=''); show_shape(inputs, x)
	x = ZeroPadding2D(padding=(0, 1, 0, 1), dim_ordering='default')(x)
	pf("ZeroPadding2D(0,1,0,1): ", sep='', end=''); show_shape(inputs, x)

	x = Convolution2D(1, img_width, img_height, activation=convact, border_mode='same', subsample=(1,1))(x)
	pf("conv2d(", 1, ",img_width,img_height): ", sep='', end=''); show_shape(inputs, x)

	#exit(0)


	model = Model(input=inputs, output=x)
	pf(model.summary())
	pf("final prediction: ", sep='', end=''); show_shape(inputs, x)
	#exit(0)
	pf("Compiling model")
	model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
#	pf("Loading weights")
#	if load_weights and isfile(weight_store_imgdata):
#		model.load_weights(weight_store_imgdata)
#	if load_weights and isfile(weight_store_angles):
#		model_train_angles.load_weights(weight_store_angles)

	#predictions = Dense(10, activation='softmax')(x)

	# Old
	#model = Sequential()
	##model.add(
		#MaxPooling2D(pool_size=(2, 2), strides=None,
			#border_mode='valid', dim_ordering='default'))
	#model.add(Flatten())
	#model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
	#model.add(Dense(8, init='uniform', activation='relu'))
	pf("Returning model")
	return model

def view_img(label, img, show=False):
	global axs
	global fig
	global whichsubplot
	global show_images
	pf("View img")
	#img = mpimg.imread('stinkbug.png')
	plotrows = 5
	plotcols = 4
	#show_images=0
	if not show_images: return
	#pf(img)
	whichsubplot += 1
	if (whichsubplot >= plotrows*plotcols):
		whichsubplot = 0
	#pf(fig)
	if fig is None:
		fig,axs = plt.subplots(plotrows,plotcols,figsize=(5,6))
		plt.ion()
		plt.pause(0.05) # Calls matplotlib's event loop
		fig.subplots_adjust(hspace=0)
		fig.subplots_adjust(wspace=.3)
	#pf(fig)
	#pf("  Plotting whichsubplot:", whichsubplot)
	yax = int(whichsubplot/plotcols)
	xax = int(whichsubplot % plotcols)
	#pf("  Current axes (y,x):", yax, xax)
	#pf("  Deleting...")
	fig.delaxes(axs[yax][xax]) 
	#pf("  Making new subplot...")
	newaxs = plt.subplot(plotrows, plotcols, whichsubplot+1)
	axs[yax][xax] = newaxs
	plt.title(label, fontsize=10)
	plt.axis('off')
	axs[yax][xax].imshow(img, cmap="gray", interpolation='nearest') # interpolation='None'
	plt.pause(0.05) # Calls matplotlib's event loop
	#plt.colorbar()
	#plt.axes.get_xaxis().set_visible(False)
	#plt.axes.get_yaxis().set_visible(False)
	#if show:
		#plt.show()
		#plt.pause(0.05) # Calls matplotlib's event loop
	pf("/view img")

def get_rand_sampling(array, count):
	#pf("rand array:", array)
	alen = len(array)
	ret = []
	for i in range(0, count):
		ret.append(array[randint(0,alen-1)])
	#pf("Rand subset:", ret)
	return ret
	
def get_random_imgid_bundles(imgs, unique=1, ideal=1, bent=1):
	inps=[]
	outs=[]
	if ideal > 1:
		pf("get_random_imgid_bundles(): Error: ideal count can only be 1 for now")
		exit(0)
	pf("Loading image bundles (", unique*bent, ")", sep="");
	id_set = get_rand_sampling(imgs, unique)
	pf("Loading img ids (count:", len(id_set))
	#exit(0)
	for imgid in id_set:
		ideal_imgs = imgset_ideal(imgid)
		bent_imgs = imgset_bent(imgid)

		bent_subset = get_rand_sampling(bent_imgs, bent)
		#pf("Bent:", bent_subset)

		# We only have one ideal image per imgid right now, so we repeat it 10 times
		ideal_subset = [ideal_imgs[0]] * bent

		outs.extend(ideal_subset)
		inps.extend(bent_subset)
	#for i in range(0, len(inps)):
		#pf("[",i,"] ", inps[i], " -> ", outs[i], sep='')
	pf("/Loading image bundles")
	return inps, outs

def randdeform(img, xoffset=0, yoffset=0, fill=0):  # img:numpy array (1, w, h)
	#img=mpimg.imread('stinkbug.png')
	#img = np.array(list(itertools.chain(*[range(8)]*8))).reshape((1,8,8))
	#pf("stinkbug shape:", img.shape)
	#exit(0)
	if xoffset >= 1 or yoffset >= 1 or xoffset < 0 or yoffset < 0:
		pf("Please don't call randdeform() with offsets outside of [0,1)")
		exit(0)
	w = img.shape[1]
	h = img.shape[2]
	xoff = int(w * xoffset)
	xoff = randint(-xoff, xoff)
	yoff = int(w * yoffset)
	yoff = randint(-yoff, yoff)
	#view_img("Orig", img[0], show=True)
	#np.set_printoptions(threshold=64, linewidth=termwidth-1, edgeitems=10)
	#pf("IMG-roll:"); pf(img)
	#img = np.array([[1,2,3,4,5],[6,7,8,9,0]])
	#img = img.reshape((1,) + img.shape)
	#img.transpose(1,2,0)
	#pf("Image shape:", img.shape)
	#time.sleep(3)
	#view_img("No roll", img, show=True)
	if xoff:
		img = np.roll(img, xoff, axis=2);
		if xoff > 0:
			img[:,:,:xoff].fill(fill)
		else:
			img[:,:,xoff:].fill(fill)
	if yoff:
		img = np.roll(img, yoff, axis=1);
		if yoff > 0:
			img[:,:yoff,:].fill(fill)
		else:
			img[:,yoff:,:].fill(fill)
	#pf("Rolling x:", xoff, " y:", yoff, sep='')
	#view_img("Roll", img[0], show=True)
	#time.sleep(10);
	#exit(0)
	#time.sleep(15)
	#, axis=1) # axis=1 = xaxis
	return img
	
def imgids_to_imgs(imgids, deform='small'):
	if deform == 'none':
		offset = 0.0
	elif deform == 'small':
		offset = .06  # 2/32 pixels right now
	else:
		offset = .12
	iset=[]
	for imgid in imgids:
		img = load_img(imgid, grayscale='True')
		img = img_to_array(img)  # Numpy array with shape (1, width, height)
		if normalize:
			img = (img/255.0) - .5   # I guess we want -1 .. 1
		img = randdeform(img, xoffset=offset, yoffset=offset, fill=0)
		#img = img.reshape((1,) + img.shape)  # Numpy array with shape (1, 1, w, h)
		#pf("Image", imgid, "Shape:", img.shape)
		iset.append(img)
	iset = np.array(iset)
	pf("iset shape:", iset.shape)
	return iset

def test_imgs(count=-1):
	iteration = 0
	pf("Images in img_ids_test", len(img_ids_test))
	pf(img_ids_test)
	while count < -1 or iteration < count:
		iteration += 1
		valximgids,valyimgids = get_random_imgid_bundles(img_ids_test, unique=5, ideal=1, bent=1)
		xval=imgids_to_imgs(valximgids, deform='small')
		yval=imgids_to_imgs(valyimgids, deform='none')
		bsize = len(xval)
		pf("xval shape", xval.shape)
	
		# evaluate the model
		scores = model.evaluate(xval, yval, batch_size=bsize, verbose=1)
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		prediction = model.predict(xval, batch_size=bsize, verbose=1)
		for i in range(0, 0):
			view_img("(VIn"+str(i)+")", xval[i][0], show=True)
			view_img("(VPred)", prediction[i][0])

def view_weights(model, name=None):
	if name == None:
		raise ValueError("Call with layer")
	layer = model.get_layer(name)
	weights = layer.get_weights()
	pf(weights[1])

def train_angles(model, loopsets, viewskip, epochs):
	# For every src_img_bundle (unique set of word-pages), we create a number
	# of pairs (one for each bent_img_bundle).
	# Really it's for every ideal_img_bundle too, but we currently make only one
	# ideal image.  Nevertheless, it results in bundles of size:
	# src_img_bundle * ideal_img_bundle * bent_img_bundle
	src_img_bundle=10    # Img IDs correspond to sets of words on pages, each with
	                     #  some number of ideal flat images (only 1 right now), and
						 #  some number of bent images
	ideal_img_bundle=1   # We only have 1 flat page right now
	bent_img_bundle=95   # Bunch of these
	train_epochs=epochs
	valfrac=.3           # When used (it's not right now), this is the fraction of
	                     # the set to reserve for validating the model's results
	iterations = 0
	total_train = 0
	valximgids,valyimgids = get_random_imgid_bundles(img_ids_test, 10, 1, 1)
	xval=imgids_to_imgs(valximgids, deform='small')
	yval=imgids_to_imgs(valyimgids, deform='none')

	while iterations < loopsets:
		iterations += 1
		ximgids,yimgids = get_random_imgid_bundles(img_ids_train, src_img_bundle, ideal_img_bundle, bent_img_bundle)
		bsize=len(ximgids)
		pf("Bundle size:", bsize)
		#exit(0)
		x=imgids_to_imgs(ximgids, deform='large')
		y=imgids_to_imgs(yimgids, deform='none')

			# Validation set uses equal count of bent pages to source (ideal) images
		#valximgids,valyimgids = get_random_imgid_bundles(img_ids_test, int(ceil(src_img_bundle*valfrac)), int(ceil(ideal_img_bundle*valfrac)), 1)
		#xval=imgids_to_imgs(valximgids, deform='small')
		#yval=imgids_to_imgs(valyimgids, deform='none')

		#pf(len(ximgids), len(yimgids))
		#pf(len(valximgids), len(valyimgids))
		#exit(0)
		pf("model.fit() batchsize(", bsize, ")*epochs(", train_epochs, ") = ", bsize*train_epochs, sep='')

		#pf("Viewing intermediates:")
		#view_angles = model_view_angles.predict(x, batch_size=bsize, verbose=1)
		#view_img("ViewAng", view_angles[0][0])
		#view_angles = model_view_imgdata.predict(x, batch_size=bsize, verbose=1)
		#view_img("ViewIMG", view_angles[0][0])
		#view_weights(model, name="imgdata")
		#pre_weights = model.get_layer("imgdata").get_weights()[1]
		history = model.fit(x, y, validation_data=(xval,yval), batch_size=bsize, nb_epoch=train_epochs, verbose=1)
		#post_weights = model.get_layer("imgdata").get_weights()[1]
		pf("/model.fit()")
		#view_weights(model, name="imgdata")
		#pf(pre_weights)
		#pf(post_weights)
		#exit(0)

		total_train += bsize*train_epochs
		#if not (total_train % (bsize*10)):

		#pf("Viewing intermediates:")
		#view_angles = model_view_angles.predict(x, batch_size=bsize, verbose=1)
		#view_img("ViewAng", view_angles[0][0])
		#view_angles = model_view_imgdata.predict(x, batch_size=bsize, verbose=1)
		#view_img("ViewIMG", view_angles[0][0])

		if not iterations % viewskip:
			pf("Total trainings:", total_train)
			pf("Predicting:")
			prediction = model.predict(x, batch_size=bsize, verbose=1)
			pf("/Predicting:")
			pf("Displaying input image [0]")
			pf("Shape of image we're about to display", y[0][0].shape)
			#time.sleep(5)
			view_img("(AngIn)", x[0][0], show=True)
			view_img("(AngOut)", y[0][0], show=True)
			#view_img("(IVal)", xval[0][0], show=True)
			#view_img("(OVal)", yval[0][0], show=True)
			#view_img("Bent (Input)", x[0][0], show=True)
			pf("Displaying prediction image [0][0]")
			pf(prediction[0][0])
			pf("Pred min: ", prediction[0][0].min())
			pf("Pred max: ", prediction[0][0].max())
			#plt.hist(prediction[0][0], bins=256, range=(0.0, 1.0))
			view_img("Ang #"+str(total_train), prediction[0][0])
			test_imgs(2)
			plt.show()
			#exit(0)
			pf("/Displaying prediction image")

def train_bundles(model, loopsets, viewskip, epochs):
	iterations = 0
	total_train = 0
	src_img_bundle=40    # Img IDs correspond to sets of words on pages, each with
	                     #  some number of ideal flat images (only 1 right now), and
						 #  some number of bent images
	ideal_img_bundle=1   # We only have 1 flat page right now
	bent_img_bundle=45   # Bunch of these
	train_epochs=epochs
	valfrac=.5
	while iterations < loopsets:
		iterations += 1
		ximgids,yimgids = get_random_imgid_bundles(img_ids_train, unique=src_img_bundle, ideal=ideal_img_bundle, bent=bent_img_bundle)
		bsize=len(ximgids)
		pf("Bundle size:", bsize)
		#exit(0)
		x=imgids_to_imgs(ximgids, deform='large')
		y=imgids_to_imgs(yimgids, deform='none')

			# Validation set uses equal count of bent pages to source (ideal) images
		valximgids,valyimgids = get_random_imgid_bundles(img_ids_test, unique=int(ceil(src_img_bundle*valfrac)), ideal=int(ceil(ideal_img_bundle*valfrac)), bent=1)
		xval=imgids_to_imgs(valximgids, deform='small')
		yval=imgids_to_imgs(valyimgids, deform='none')

		#pf(len(ximgids), len(yimgids))
		#pf(len(valximgids), len(valyimgids))
		#exit(0)
		pf("model.fit() batchsize(", bsize, ")*epochs(", train_epochs, ") = ", bsize*train_epochs, sep='')
		#pre_weightsa = model.get_layer("angles").get_weights()[1]
		#pre_weightsi = model.get_layer("imgdata").get_weights()[1]
		history = model.fit(x, y, validation_data=(xval,yval), batch_size=bsize, nb_epoch=train_epochs, verbose=1)
		#post_weightsa = model.get_layer("angles").get_weights()[1]
		#post_weightsi = model.get_layer("imgdata").get_weights()[1]
		#pf("/model.fit()")
		#pf("Angle weights")
		#pf(pre_weightsa)
		#pf(post_weightsa)
		#pf("ImgData weights")
		#pf(pre_weightsi)
		#pf(post_weightsi)
		#exit(0)
		total_train += bsize*train_epochs
		#if not (total_train % (bsize*10)):
		if not iterations % viewskip:
			pf("Total trainings:", total_train)
			pf("Predicting:")
			prediction = model.predict(x, batch_size=bsize, verbose=1)
			pf("/Predicting:")
			pf("Displaying input image [0]")
			pf("Shape of image we're about to display", y[0][0].shape)
			#time.sleep(5)
			view_img("(B Inp)", x[0][0], show=True)
			view_img("(B Out)", y[0][0], show=True)
			#view_img("(IVal)", xval[0][0], show=True)
			#view_img("(B OVal)", yval[0][0], show=True)
			#view_img("Bent (Input)", x[0][0], show=True)
			pf("Displaying prediction image [0][0]")
			pf(prediction[0][0])
			pf("Pred min: ", prediction[0][0].min())
			pf("Pred max: ", prediction[0][0].max())
			#plt.hist(prediction[0][0], bins=256, range=(0.0, 1.0))
			view_img("B pred #"+str(total_train), prediction[0][0])
			plt.show()
			#exit(0)
			pf("/Displaying prediction image")

def train_nn(model):
	x = None
	y = None
	pf("Training...")
	total_train = 0
	total_batches = 0
	dest_count = 0
	#for our_epochs in range(max_main_imgloops):
		#train_id = get_rand_img_train_id()
	while True:
		for imgid in img_ids_train:
			dest_count += 1
			ideal_imgs = imgset_ideal(imgid)
			bent_imgs = imgset_bent(imgid)
			pf("Working on IMG ID:", imgid)
			for ideal in ideal_imgs:
				pf("Training dest", dest_count)
				pf("Loading ideal image:", ideal)
				img = load_img(ideal, grayscale='True')  # PIL image
				#img = img.resize((69,69), Image.ANTIALIAS)
				#pf("Input image size:", img.size)
				y = img_to_array(img)  # Numpy array with shape (1, 150, 150)
				#pf("Ideal image:")
				#pf(y)
				#view_img("Orig Input", y[0])
				if normalize:
					y = (y/255.0) - .5
				#pf("Ideal image regularized:")
				#pf(y)
				#exit(0)
				#pf(y.shape)
				#view_img(img)
				#pf("Output Image")
				y = y.reshape((1,) + y.shape)  # Numpy array with shape (1, 1, 150, 150), I think so it's a batch of count(1 right now),channels,rows,cols
				#pf("Output Image")
				#pf(y)
				j = 0
				datagen_output.fit(y)
				for out_batch in datagen_output.flow(y, batch_size=3):
					pf("  Batch:", dest_count, "->", j)
					j += 1
					if j > out_batch_versions:
						break

					for bent in get_rand_sampling(bent_imgs, 2):
						#pf("    Bent:", dest_count, "->", j)
						pf("Trained:", total_train, "Loading bent image:", bent)
						img = load_img(bent, grayscale='True')  # PIL image
						x = img_to_array(img)  # Numpy array with shape (1, 150, 150)
						if normalize:
							x = (x/255.0) - .5
						#pf("     Bent image:")
						#pf(x)
						x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 3, 150, 150)
						i = 0
						bsize = 3
						datagen_input.fit(x)
						for in_batch in datagen_input.flow(x, batch_size=bsize):
							#pf("      Batch:", i, "of dest count", dest_count)
							#plt.hist(in_batch.flatten())
							#plt.show()
							i += 1
							if i > in_batch_versions:
								break  # otherwise the generator would loop indefinitely
							#pf(in_batch)
							#pf("-- Fitting ----------------\n")
							#pf("InBatch:")
							#pf(in_batch.shape)
							#pf("/InBatch:")
							#pf("OutBatch:")
							#pf(out_batch.shape)
							#pf("/OutBatch:")
							#pf("Fitting:")
							history = model.fit(in_batch, out_batch, batch_size=bsize, nb_epoch=train_epochs, verbose=0)
							#pf("/Fitting:")
							total_train += bsize
							total_batches += 1
							#pf("-- Fitting History --------\n")
							#pf(history.history)
							if not (total_batches % 100):
								pf("Trained: {}".format(total_train))
							if total_train == 1 or not (total_train % 200):
								pf("Predicting:")
								prediction = model.predict(x)
								pf("/Predicting:")
								#pf(prediction[0][0])
								#pf("Image Shape")
								#pf(prediction[0][0].shape)
								pf("Current destinations trained:", dest_count)
								pf("Displaying input image [0]")
								#pf(in_batch[0][0].shape)
								#pf(in_batch[0][0])
								#pf(prediction[0][0].shape)
								#pf(prediction[0][0])
								view_img("Ideal (Output)", y[0][0])
								view_img("Bent (Input)", x[0][0])
								view_img("ibatch[0]", in_batch[0][0])
								pf("Displaying prediction image [0][0]")
								pf(prediction[0][0])
								pf("Pred min: ", prediction[0][0].min())
								pf("Pred max: ", prediction[0][0].max())
								#plt.hist(prediction[0][0], bins=256, range=(0.0, 1.0))
								view_img("Train pred ("+str(total_train)+")", prediction[0][0])
								plt.show()
								#exit(0)
								pf("/Displaying prediction image")
								#exit(0)
							#pf(prediction)
							#exit(0)
							#scores = model.evaluate(in_batch,out_batch)
							#pf("\033[33;1m%s: %.2f%%\033[0m" % (model.metrics_names[1], scores[1]*100))

def save_weight_sets():
	if save_weights:
		model.save_weights(weight_store_imgdata)
		model_train_angles.save_weights(weight_store_angles)
		pf("Saved weights.")

datagen_input, datagen_output = init()
load_imgnames()
model, model_train_angles, model_view_angles, model_view_imgdata = create_nn2()
if opt_run_test:
	test_imgs()
	exit(0)

runs=0
ang_epochs=1
bund_epochs=1
train_angles(model_train_angles, loopsets=2, viewskip=1, epochs=1)
save_weight_sets()
while True:
	runs += 1
	pf("=========================================================================")
	pf("Runs:", runs, " Angle-epochs:", ang_epochs, " Imgdata-epochs:", bund_epochs)
	train_bundles(model, loopsets=1, viewskip=1, epochs=bund_epochs)
	save_weight_sets()
	train_angles(model_train_angles, loopsets=1, viewskip=1, epochs=ang_epochs)
	save_weight_sets()

	#ang_epochs = int(math.tanh(runs/100)*1000+150)
	
sys.exit(0)


# vim:ts=4 ai
