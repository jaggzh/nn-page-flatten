#!/usr/bin/python
from __future__ import print_function # For eprint
from keras.models import Sequential, Model # , load_weights, save_weights
from keras.layers import Dense, merge, Reshape, UpSampling2D, Flatten, Convolution2D, MaxPooling2D, Input, ZeroPadding2D, Activation, Dropout
#from keras.layers import Deconvolution2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.layer_utils import print_summary
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
import keras.initializations as inits
#from STL.SpatialTransformer import *
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
from keras.callbacks import EarlyStopping
import shutil
from keras import backend as K
#from seya.layers.attention import SpatialTransformer, ST2

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

input_shape = None
weight_store_imgdata = "weights-imgdata.h5"
weight_store_angles = "weights-angles.h5"
weight_store_angles_minloss = "weights-angles-minloss.h5"
bent_dir = "blend/pages-64"
ideal_dir = "blend/ideal-64"
img_ids_train=[]
img_ids_valid=[]
img_ids_test=[]
verbose=0
show_images=1
dim = 64
indimx = indimy = 64
img_width=dim
img_height=dim
max_imagesets=1000 # imageset = Each unique page of words (bent or flat) (-1 is unlimited)
#max_main_imgloops = 100 # Number of times to loop through entire set of images
train_epochs=1
out_batch_versions=3 # number of distorted images to feed in
in_batch_versions=3 # number of distorted images to feed in
load_weights=1      # load prior run stored weights
save_weights=1      # load prior run stored weights
test_fraction = .07  # fraction of the data set for the test set
valid_fraction = .07  # fraction of the data set for the validation set
whichsubplot = -1
axs = None
fig = None
normalize = True
#opt_run_test = True
opt_run_test = False
early_stopping_ang = None
early_stopping_img = None
checkpoint_low_loss = None

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
			r = random.random()
			if r < test_fraction:
				img_ids_test.append(name)
			elif r < test_fraction + valid_fraction:
				img_ids_valid.append(name)
			else:
				img_ids_train.append(name)
		if max_imagesets > 0 and i > max_imagesets:
			break
def init():
	# fix random seed for reproducibility
	global termwidth, termheight
	global early_stopping_ang
	global early_stopping_img
	global checkpoint_low_loss
	termwidth, termheight = get_linux_terminal()
	seed = 12
	random.seed(seed)
	np.random.seed(seed)
	np.set_printoptions(threshold=64, linewidth=termwidth-1, edgeitems=3)
#	datagen_input = ImageDataGenerator(
#		rotation_range=0,
#		width_shift_range=0.09,
#		height_shift_range=0.09,
#		shear_range=0,
#		zoom_range=0.1,
#		horizontal_flip=False,
#		fill_mode='nearest')
#		#featurewise_center=True,
#		#featurewise_std_normalization=True)
#	datagen_output = ImageDataGenerator(
#		rotation_range=0,
#		width_shift_range=0.09,
#		height_shift_range=0.09,
#		shear_range=0,
#		zoom_range=0.1,
#		horizontal_flip=False,
#		fill_mode='nearest')
#		#featurewise_center=True,
#		#featurewise_std_normalization=True)
	early_stopping_ang = EarlyStopping(monitor='val_loss', patience=10)
	early_stopping_img = EarlyStopping(monitor='val_loss', patience=10)
	checkpoint_low_loss = ModelCheckpoint(filepath=weight_store_angles_minloss, verbose=1, save_best_only=True, save_weights_only=True)
	#return datagen_input, datagen_output
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

def stn_prep_data():
	global input_shape
	global x_train, y_train, x_valid, y_valid, x_test, y_test
	ximgids,yimgids = make_imgid_bundles(img_ids_train)
	valximgids,valyimgids = make_imgid_bundles(img_ids_valid)
	testximgids,testyimgids = make_imgid_bundles(img_ids_test)

	x_train=imgids_to_imgs(ximgids, deform='small')
	y_train=imgids_to_imgs(yimgids, deform='none')
	x_valid=imgids_to_imgs(valximgids, deform='small')
	y_valid=imgids_to_imgs(valyimgids, deform='none')
	x_test=imgids_to_imgs(testximgids, deform='small')
	#y_test=imgids_to_imgs(testyimgids, deform='none')
	y_test=np.zeros(x_test.shape)
	input_shape = x_train.shape[1:]

def stl_matrix():
	w = np.zeros((20, 6))
	b = np.zeros((6,))
	b[0] = 1.
	b[4] = 1.
	return [w,b]
	#b = np.zeros((2, 3), dtype='float32')
	#b[0, 0] = 1
	#b[1, 1] = 1
	#W = np.zeros((50, 6), dtype='float32')
	#weights = [W, b.flatten()]
def create_nn_stn():
	global input_shape
	global x_train, y_train, x_valid, y_valid, x_test, y_test
	act='tanh'

#	locnet = Input(shape = (1, img_width, img_height))
#	locnet = MaxPooling2D(pool_size=(2,2), input_shape=input_shape)(locnet) # 64 -> 32
#	locnet = Convolution2D(20, 5, 5)(locnet)
#	locnet = MaxPooling2D(pool_size=(2,2))(locnet) # -> 16
#	locnet = Convolution2D(20, 5, 5)(locnet)
#	locnet = MaxPooling2D(pool_size=(2,2))(locnet) # -> 8
#	locnet = Convolution2D(20, 5, 5)(locnet)
#	locnet = Flatten()(locnet)
#	locnet = Dense(50)(locnet)
#	locnet = Activation('relu')(locnet)
#	locnet = Dense(6, weights=weights)(locnet)
	#	#locnet.add(Activation('sigmoid'))
	input = Input(shape=(3, indimy, indimx))
	# 64
	x = Convolution2D(32, 3, 3)(x)

	x = MaxPooling2D(pool_size=(2,2))(x)
	# 32
	x = Convolution2D(80, 3, 3)(x)

	x = MaxPooling2D(pool_size=(2,2))(x)
	# 16
	x = Convolution2D(160, 3, 3)(x)

	x = MaxPooling2D(pool_size=(2,2))(x)
	# 8
	x = Convolution2D(20, 3, 3)(x)

	x = MaxPooling2D(pool_size=(2,2))(x)
	# 4
	x = Convolution2D(20, 3, 3)(x)
	x = Flatten()(x)

	x = Dense(10)(x)
	matrix = Dense(6, weights=stl_matrix())(x)
	trans = SpatialTransformerLayer(downsample_factor=1.0)([input, x])

	model = Model(input=input, output=[trans, matrix])

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	#model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
	return model

def train_stn(model, imgsets, bentvers, loopsets, viewskip, epochs):
	#XX = model.get_input()
	#YY = model.layers[0].get_output()
	#F = theano.function([XX], YY)
	global x_train, y_train, x_valid, y_valid, x_test, y_test

	nb_epochs = 1000 # you probably want to go longer than this
	batch_size = 256
	#try:
	for e in range(nb_epochs):
		print('-'*40)
		#progbar = generic_utils.Progbar(x_train.shape[0])
		for b in range(x_train.shape[0]/batch_size):
			f = b * batch_size
			l = (b+1) * batch_size
			x_batch = x_train[f:l] # .astype('float32')
			y_batch = y_train[f:l] # .astype('float32')
			loss = model.train_on_batch(x_batch, y_batch)
			#progbar.add(x_batch.shape[0], values=[("train loss", loss)])
		#scorev = model.evaluate(x_valid, y_valid, show_accuracy=True, verbose=0)[1]
		#scoret = model.evaluate(x_test, y_test, show_accuracy=True, verbose=0)[1]
		scorev = model.evaluate(x_valid, y_valid, verbose=1)[1]
		scoret = model.evaluate(x_test, y_test, verbose=1)[1]
		y_test = model.predict(x_test, batch_size=x_test.shape[0], verbose=1)
		print('Epoch: {0} | Valid: {1} | Test: {2}'.format(e, scorev, scoret))

		if e % 50 == 0:
			#xresult = F(x_batch[:9])
			#for i in range(9):
			#	view_img(Xresult[i, 0])
			#xresult = y_batch[i][0]
			view_img("x"+str(e), x_test[0][0])
			view_img("y"+str(e), y_test[0][0])
	#except KeyboardInterrupt:
	#	pass


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
	plotcols = 6
	#show_images=0
	if not show_images: return
	#pf(img)
	whichsubplot += 1
	if (whichsubplot >= plotrows*plotcols):
		whichsubplot = 0
	#pf(fig)
	if fig is None:
		fig,axs = plt.subplots(plotrows,plotcols,figsize=(10,11))
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
	
def make_imgid_bundles(imgs, unique=1, ideal=1, bent=1):
	inps=[]
	outs=[]
	if ideal > 1:
		pf("get_random_imgid_bundles(): Error: ideal count can only be 1 for now")
		exit(0)
	#pf("Loading image bundles (", unique*bent, ")", sep="");
	id_set = imgs
	#pf("Loading img ids (count:", len(id_set))
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
	#pf("/Loading image bundles")
	return inps, outs

def get_random_imgid_bundles(imgs, unique=1, ideal=1, bent=1):
	inps=[]
	outs=[]
	if ideal > 1:
		pf("get_random_imgid_bundles(): Error: ideal count can only be 1 for now")
		exit(0)
	#pf("Loading image bundles (", unique*bent, ")", sep="");
	id_set = get_rand_sampling(imgs, unique)
	#pf("Loading img ids (count:", len(id_set))
	#exit(0)
	for imgid in id_set:
		ideal_imgs = imgset_ideal(imgid)
		bent_imgs = imgset_bent(imgid)
		#bent_imgs = imgset_ideal(imgid)

		bent_subset = get_rand_sampling(bent_imgs, bent)
		#pf("Bent:", bent_subset)

		# We only have one ideal image per imgid right now, so we repeat it 10 times
		ideal_subset = [ideal_imgs[0]] * bent

		outs.extend(ideal_subset)
		inps.extend(bent_subset)
	#for i in range(0, len(inps)):
		#pf("[",i,"] ", inps[i], " -> ", outs[i], sep='')
	#pf("/Loading image bundles")
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
		offset = .03  # 2/64 pixels right now
	else:             # large
		offset = .12  # 64*.12
	iset=[]
	for imgid in imgids:
		img = load_img(imgid, grayscale='True')
		img = img_to_array(img)  # Numpy array with shape (1, width, height)
		if normalize:
			img = (img/255.0)
		img = randdeform(img, xoffset=offset, yoffset=offset, fill=0)
		#img = img.reshape((1,) + img.shape)  # Numpy array with shape (1, 1, w, h)
		#pf("Image", imgid, "Shape:", img.shape)
		iset.append(img)
	iset = np.array(iset)
	#pf("iset shape:", iset.shape)
	return iset

def test_imgs(count=1):
	iteration = 0
	pf("Images in img_ids_test", len(img_ids_test))
	pf(img_ids_test)
	while iteration < count:
		iteration += 1
		ximgids,yimgids = get_random_imgid_bundles(img_ids_test, unique=5, ideal=1, bent=1)
		xval=imgids_to_imgs(ximgids, deform='small')
		yval=imgids_to_imgs(yimgids, deform='none')
		bsize = len(xval)
		pf("xval shape", xval.shape)
	
		# evaluate the model
		scores = model.evaluate(xval, yval, batch_size=bsize, verbose=1)
		print("Metric names: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		prediction = model.predict(xval, batch_size=bsize, verbose=1)
		for i in range(0, 1):
			view_img("(VIn"+str(i)+")", xval[i][0], show=True)
			view_img("(VPred)", prediction[i][0])

def view_weights(model, name=None):
	if name == None:
		raise ValueError("Call with layer")
	layer = model.get_layer(name)
	weights = layer.get_weights()
	pf(weights[1])

def train_angles(model, imgsets, bentvers, loopsets, viewskip, epochs):
	global early_stopping_ang
	global checkpoint_low_loss
	# For every src_img_bundle (unique set of word-pages), we create a number
	# of pairs (one for each bent_img_bundle).
	# Really it's for every ideal_img_bundle too, but we currently make only one
	# ideal image.  Nevertheless, it results in bundles of size:
	# src_img_bundle * ideal_img_bundle * bent_img_bundle
	src_img_bundle=imgsets  # Img IDs correspond to sets of words on pages, each with
	                     #  some number of ideal flat images (only 1 right now), and
						 #  some number of bent images
	ideal_img_bundle=1   # We only have 1 flat page right now
	bent_img_bundle=bentvers   # Bunch of these for each of src_img_bundle
	train_epochs=epochs
	valfrac=.3           # When used (it's not right now), this is the fraction of
	                     # the set to reserve for validating the model's results
	iterations = 0
	total_train = 0
	valximgids,valyimgids = get_random_imgid_bundles(img_ids_test, 80, 1, 1)
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

		#view_weights(model, name="imgdata")
		#pre_weights = model.get_layer("imgdata").get_weights()[1]
		#history = model.fit(x, y, validation_data=(xval,yval), batch_size=bsize, nb_epoch=train_epochs, verbose=1, callbacks=[early_stopping])

		history = model.fit(x, y, validation_data=(xval,yval), batch_size=bsize, nb_epoch=train_epochs, verbose=1, callbacks=[checkpoint_low_loss, early_stopping_ang])
		#post_weights = model.get_layer("imgdata").get_weights()[1]
		pf("/model.fit()")
		#view_weights(model, name="imgdata")
		#pf(pre_weights)
		#pf(post_weights)
		#exit(0)

		total_train += bsize*train_epochs

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

def create_nn2():
	act='tanh'

	inputs = Input(shape = (1, img_width, img_height))
	x = inputs

	# 64
	x = Dropout(.05)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	d64 = x

	x = MaxPooling2D((2,2), border_mode='same', dim_ordering='th')(x)
	# 32
	x = Dropout(.05)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	d32 = x
	x = MaxPooling2D((2,2), border_mode='same', dim_ordering='th')(x)
	# 16
	x = Dropout(.05)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	d16 = x
	x = MaxPooling2D((2,2), border_mode='same', dim_ordering='th')(x)
	# 8
	x = Dropout(.01)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	#x = MaxPooling2D((2,2), border_mode='same', dim_ordering='th')(x)
	# 4
	x = Flatten()(x)
	x = Dense(20)(x)
	x = Dense(64)(x)
	#x = Reshape((1,4,4))(x)

	#x = Convolution2D(256, 2, 2, activation='relu', border_mode='same', subsample=(1,1))(x)
	## 4
	#x = UpSampling2D(size=(2,2), dim_ordering='th')(x)

	transform_params = Dense(64, name="xform", init='zero')(x)
	transform_params = Reshape((1,8,8))(transform_params)

	x = Reshape((1,8,8))(x)
	# 8
	x = merge([x,transform_params], mode='concat', concat_axis=1)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)

	x = UpSampling2D(size=(2,2), dim_ordering='th')(x)
	# 16
	d16 = Dropout(.2)(d16)
	x = merge([x, d16], mode='concat', concat_axis=1)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)

	x = UpSampling2D(size=(2,2), dim_ordering='th')(x)
	# 32
	d32 = Dropout(.2)(d32)
	x = merge([x, d32], mode='concat', concat_axis=1)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)

	x = UpSampling2D(size=(2,2), dim_ordering='th')(x)
	# 64
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	x = Convolution2D(32, 2, 2, border_mode='same')(x); x = LeakyReLU(alpha=.2)(x)
	d64 = Dropout(.2)(d64)
	x = merge([x, d64], mode='concat', concat_axis=1)
	x = Convolution2D(1, 1, 1, border_mode='same')(x);  x = LeakyReLU(alpha=.2)(x)
	x = Dropout(.01)(x)
	x = Convolution2D(1, 1, 1, activation='tanh', border_mode='same', subsample=(1,1))(x)

	pf("Compiling models")
	#joined = merge([angles_called, imgdata_called], mode='concat', concat_axis=1)
	model_xform_frozen = Model(input=inputs, output=x)
	freeze_layers(model_xform_frozen, names=['xform'])
	opt=Adam(lr=0.000025, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model_xform_frozen.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

	model_xform_only = Model(input=inputs, output=x)
	freeze_layers(model_xform_only, names=['xform'], invert=1)
	freeze_layers(model_xform_only, names=['xform'], unfreeze=1)
	opt=Adam(lr=0.00005, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model_xform_only.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

	pf("\033[36;1mFrozen xform layer:\033[0m")
	pf(model_xform_frozen.summary())
	pf("\033[32;1mFrozen ALL but xform layer:\033[0m")
	pf(model_xform_only.summary())

	#pf("final prediction: ", sep='', end=''); show_shape(inputs, x)

	#sgd=SGD(lr=0.1, momentum=0.000, decay=0.0, nesterov=False)


	#model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

	pf("Loading weights")
	if load_weights and isfile(weight_store_imgdata):
		model_xform_frozen.load_weights(weight_store_imgdata)
	return model_xform_frozen, model_xform_only

def freeze_layers(model, names=[], indexes=[], unfreeze=False, invert=False):
	un_msg = "Un-" if unfreeze else ""     # Marking untrainable or not
	un_val = True if unfreeze else False   # Trainable or not
	layers=model.layers
	for i in range(len(layers)):
		name=layers[i].name
		layer=layers[i]
		if not invert and (i in indexes):
			pf(un_msg, "Freezing layer [", i, "] by index. Name=", name, sep='')
			layer.trainable = un_val;
		elif not invert and (name in names):
			pf(un_msg, "Freezing layer [", i, "] by name. Name=", name, sep='')
			layer.trainable = un_val;
		elif invert and (not (name in names) and not (i in indexes)):
			pf(un_msg, "Freezing layer [", i, "]. Name=", name, sep='')
			layer.trainable = un_val;

total_train = 0
def train_bundles(model=None, imgsets=1, bentvers=1, loopsets=1, viewskip=1, epochs=1):
	global early_stopping_img
	global checkpoint_low_loss
	global total_train
	iterations = 0
	src_img_bundle=imgsets  # Img IDs correspond to sets of words on pages, each with
	                     #  some number of ideal flat images (only 1 right now), and
						 #  some number of bent images
	ideal_img_bundle=1   # We only have 1 flat page right now
	bent_img_bundle=bentvers   # Bunch of these
	train_epochs=epochs
	valfrac=.2

	while iterations < loopsets:
		iterations += 1
		ximgids,yimgids = get_random_imgid_bundles(img_ids_train, unique=src_img_bundle, ideal=ideal_img_bundle, bent=bent_img_bundle)
		bsize=len(ximgids)
		pf("Bundle size:", bsize)
		#exit(0)
		x=imgids_to_imgs(ximgids, deform='none')
		y=imgids_to_imgs(yimgids, deform='none')

			# Validation set uses equal count of bent pages to source (ideal) images
		#valximgids,valyimgids = get_random_imgid_bundles(img_ids_test, unique=int(ceil(src_img_bundle*valfrac)), ideal=int(ceil(ideal_img_bundle*valfrac)), bent=1)
		#xval=imgids_to_imgs(valximgids, deform='small')
		#yval=imgids_to_imgs(valyimgids, deform='none')

		pf(len(ximgids), len(yimgids))
		#pf(len(valximgids), len(valyimgids))
		#exit(0)
		pf("model.fit() batchsize(", bsize, ")*epochs(", train_epochs, ") = ", bsize*train_epochs, sep='')
		#pre_weightsa = model.get_layer("angles").get_weights()[1]
		#pre_weightsi = model.get_layer("imgdata").get_weights()[1]
		#history = model.fit(x, y, validation_data=(xval,yval), batch_size=bsize, nb_epoch=train_epochs, verbose=1, callbacks=[checkpoint_low_loss])
		#pf("xform.weights.w:", model.get_layer(name='xform').weights[0].eval())
		#pf("xform.weights.b:", model.get_layer(name='xform').weights[1].eval())
		#pf("xform.trainable:", model.get_layer(name='xform').trainable)
		history = model.fit(x, y, batch_size=bsize, nb_epoch=train_epochs, verbose=1, callbacks=[checkpoint_low_loss])
		#pf("xform.trainable:", model.get_layer(name='xform').trainable)
		#pf("xform.weights.w:", model.get_layer(name='xform').weights[0].eval())
		#pf("xform.weights.b:", model.get_layer(name='xform').weights[1].eval())
		save_weight_sets(model=model)
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
			for i in range(min(1,len(x))):
				view_img("(Inp)", x[i][0], show=True)
				view_img("(GndTrth)", y[i][0], show=True)
				view_img("Pred #"+str(total_train), prediction[i][0])
				#view_img("(IVal)", xval[0][0], show=True)
				#view_img("(B OVal)", yval[0][0], show=True)
				#view_img("Bent (Input)", x[0][0], show=True)
				#plt.hist(prediction[0][0], bins=256, range=(0.0, 1.0))
				plt.show()
				#exit(0)
				pf("/Displaying prediction image")

def save_weight_sets(model=None):
	if save_weights:
		model.save_weights(weight_store_imgdata)
		#model_train_angles.save_weights(weight_store_angles)
		pf("Saved weights.")
def load_low_loss_weights():
	if load_weights and isfile(weight_store_angles_minloss):
		model.load_weights(weight_store_angles_minloss)

def set_layer_weights(model=None, layer=None, weights=None):
	for mlayer in (model.layers):
		if mlayer.name == layer:
			pf("Setting weights for layer: ", layer)
			mlayer.set_weights(weights)
def post_init_layer_weights(model=None, name=None, index=None, init=None):
	lay = model.get_layer(name=name, index=index)
	if not lay: raise ValueError("Model missing requested layer")
	wb = lay.get_weights()       # Weights and biases
	ww_shape = wb[0].shape
	wb_shape = wb[1].shape
	warr = np.asarray(init(shape=ww_shape, name='weights').eval())
	wbia = np.asarray(init(shape=wb_shape, name='biases').eval())
	lay.set_weights([warr, wbia])

init()
load_imgnames()
runs=0
ang_epochs=2000
bund_epochs=35
imgsets = 800
epochs = 100
stn_prep_data()
#model = create_nn_stn()
#train_stn(model, imgsets=400, bentvers=50, loopsets=1, viewskip=1, epochs=epochs)
model_xform_frozen, model_xform_only = create_nn2()
#test_imgs(10)

# Doesn't matter which model is used, since the layer weights should be shared
initial_training = 1
if initial_training:
	post_init_layer_weights(model_xform_only, name='xform', init=inits.uniform)
	pf("\033[33;1mTraining image data\033[0m")
	train_bundles(model=model_xform_frozen, imgsets=3, bentvers=10, loopsets=5, viewskip=1, epochs=100)
	
for i in range(0, 200):
	pf("\033[32;1mTraining transform data\033[0m")
	train_bundles(model=model_xform_only, imgsets=1, bentvers=30, loopsets=10, viewskip=1, epochs=200)
	pf("\033[33;1mTraining image data\033[0m")
	train_bundles(model=model_xform_frozen, imgsets=3, bentvers=10, loopsets=1, viewskip=1, epochs=100)
	#load_low_loss_weights()
	#load_low_loss_weights()
#save_weight_sets()
#pf('Press CTRL-C to quit, ENTER to copy val_loss weights to active weights...')
pf("Enter to close")
inp=raw_input('')
#if inp == "":
	#pf("Copying weights")
	#shutil.copyfile(weight_store_angles_minloss, weight_store_imgdata)
	#shutil.copyfile(weight_store_angles_minloss, weight_store_angles)
exit(0)
while False:
	runs += 1
	pf("=========================================================================")
	pf("Runs:", runs, " Angle-epochs:", ang_epochs, " Imgdata-epochs:", bund_epochs)
	train_bundles(model, loopsets=1, viewskip=1, epochs=bund_epochs)
	save_weight_sets()
	train_angles(model_train_angles, loopsets=3, viewskip=1, epochs=ang_epochs)
	save_weight_sets()

	#ang_epochs = int(math.tanh(runs/100)*1000+150)
	
# vim:ts=4 ai
