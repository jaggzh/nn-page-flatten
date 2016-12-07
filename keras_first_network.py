#!/usr/bin/python
from __future__ import print_function # For eprint
from keras.models import Sequential, Model # , load_weights, save_weights
from keras.layers import Dense, Reshape, UpSampling2D, Flatten, Convolution2D, Deconvolution2D, MaxPooling2D, Input, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.layer_utils import print_summary
import numpy as np
import sys
from os import listdir
from os.path import isdir, isfile, join
import matplotlib.pyplot as plt
from random import randint
import random
import re
import os
from PIL import Image

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

weight_store = "weights.h5"
bent_dir = "blend/pages-32"
ideal_dir = "blend/ideal-32"
img_ids_train=[]
img_ids_test=[]
verbose=0
show_images=1
datagen_input=None
datagen_output=None
img_width=32
img_height=32
max_imagesets=100 # imageset = Each unique page of words (bent or flat)
#max_main_imgloops = 100 # Number of times to loop through entire set of images
train_epochs=1
out_batch_versions=3 # number of distorted images to feed in
in_batch_versions=3 # number of distorted images to feed in
load_weights=0      # load prior run stored weights
test_fraction = .07  # Percentage (well.. fraction) of the data set for the test set
plot = None         # matplotlib image
whichsubplot = 0
plotrows = 6
plotcols = 4

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
	seed = 8
	random.seed(seed)
	np.random.seed(seed)
	np.set_printoptions(threshold=64, linewidth=termwidth-1, edgeitems=1)
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
convact='tanh'
def create_nn():
	global activation
	filters = 16

	inputs = Input(shape = (1, img_width, img_height))
	pf("input(): ", sep='', end=''); show_shape(inputs, inputs)

	x = Convolution2D(filters, 2, 2, activation=convact, border_mode='same', subsample=(1,1))(inputs)
	pf("conv2d(", filters, ",2,2): ", sep='', end=''); show_shape(inputs, x)

	x = MaxPooling2D((2,2), border_mode='same', dim_ordering='th')(x)
	pf("maxpool((2,2)): ", sep='', end=''); show_shape(inputs, x)

	filters = 8


	x = Convolution2D(filters, 2, 2, activation=convact, border_mode='same', subsample=(1,1))(x)
	pf("conv2d(", filters, ",2,2): ", sep='', end=''); show_shape(inputs, x)

	x = MaxPooling2D((2,2), border_mode='same', dim_ordering='th')(x)
	pf("maxpool((2,2)): ", sep='', end=''); show_shape(inputs, x)


	x = Convolution2D(filters, 2, 2, activation=convact, border_mode='same', subsample=(1,1))(x)
	pf("conv2d(", filters, ",2,2): ", sep='', end=''); show_shape(inputs, x)

	x = MaxPooling2D((2,2), border_mode='same', dim_ordering='th')(x)
	pf("maxpool((2,2)): ", sep='', end=''); show_shape(inputs, x)


	x = Flatten()(x) # -> 1296
	pf("flatten(): ", sep='', end=''); show_shape(inputs, x)


	x = Dense(96)(x)
	pf("dense(96): ", sep='', end=''); show_shape(inputs, x)

	x = Dense(96)(x)
	pf("dense(96): ", sep='', end=''); show_shape(inputs, x)

	x = Dense(filters*16)(x)
	pf("dense(9*filters): ", sep='', end=''); show_shape(inputs, x)

	x = Reshape((filters,4,4))(x)
	pf("reshape((", 1, ",4,4): ", sep='', end=''); show_shape(inputs, x)

	x = UpSampling2D(size = (2,2), dim_ordering='th')(x)
	pf("upsamp2d((2,2): ", sep='', end=''); show_shape(inputs, x)
	x = Deconvolution2D(filters,2,2, border_mode='same', subsample=(1,1), output_shape=(None,filters,7,7))(x)
	pf("deconv2d(", filters, ",2,2): ", sep='', end=''); show_shape(inputs, x)
	x = ZeroPadding2D(padding=(0, 1, 0, 1), dim_ordering='default')(x)
	pf("ZeroPadding2D(0,1,0,1): ", sep='', end=''); show_shape(inputs, x)

	x = UpSampling2D(size = (2,2), dim_ordering='th')(x)
	pf("upsamp2d((2,2): ", sep='', end=''); show_shape(inputs, x)
	x = Deconvolution2D(filters,2,2, border_mode='same', subsample=(1,1), output_shape=(None,filters,15,15))(x)
	pf("deconv2d(", filters, ",2,2): ", sep='', end=''); show_shape(inputs, x)
	x = ZeroPadding2D(padding=(1, 0, 1, 0), dim_ordering='default')(x)
	
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
	pf("Loading weights")
	if load_weights and isfile(weight_store):
		model.load_weights(weight_store)

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
	global plot
	global whichsubplot
	global plotrows
	global plotcols
	global show_images
	#show_images=0
	if not show_images: return
	pf(img)
	whichsubplot += 1
	if (whichsubplot > plotrows*plotcols):
		whichsubplot = 1
	if plot is None:
		plot = plt.figure(figsize=(8,11))
		plt.ion()
		plot.subplots_adjust(hspace=.5)
		plot.subplots_adjust(wspace=.4)
	plt.subplot(plotrows, plotcols, whichsubplot)
	plt.title(label, fontsize=10)
	plt.axis('off')
	fig = plt.imshow(img, cmap="gray")
	#plt.colorbar()
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)
	if show:
		plt.show()
	plot.canvas.draw()
	#plt.draw()
	#exit(0)

def get_rand_sampling(array, count):
	alen = len(array)
	ret = []
	for i in range(0, count):
		ret.append(array[randint(0,alen-1)])
	pf("Rand subset:", ret)
	return ret
	
def get_random_imgid_bundles(bundle_size):
	inps=[]
	outs=[]
	for imgid in get_rand_sampling(img_ids_train, bundle_size):
		ideal_imgs = imgset_ideal(imgid)
		bent_imgs = imgset_bent(imgid)

		bent_subset = get_rand_sampling(bent_imgs, bundle_size)

		# We only have one ideal image per imgid right now, so we repeat it 10 times
		ideal_subset = [ideal_imgs[0]] * bundle_size

		outs.extend(ideal_subset)
		inps.extend(bent_subset)
	for i in range(0, len(inps)):
		pf("[",i,"] ", inps[i], " -> ", outs[i], sep='')
	return inps, outs

def imgids_to_imgs(imgids):
	iset=[]
	for imgid in imgids:
		img = load_img(imgid, grayscale='True')
		img = img_to_array(img)  # Numpy array with shape (1, width, height)
		img = (img/255.0) - .5   # I guess we want -1 .. 1
		#img = img.reshape((1,) + img.shape)  # Numpy array with shape (1, 1, w, h)
		#pf("Image", imgid, "Shape:", img.shape)
		iset.append(img)
	iset = np.array(iset)
	#pf("iset shape:", iset.shape)
	return iset

def train_bundles(model):
	total_train = 0
	bsize=20
	while True:
		ximgids,yimgids = get_random_imgid_bundles(bsize)
		x=imgids_to_imgs(ximgids)
		y=imgids_to_imgs(yimgids)
		pf("model.fit()")
		history = model.fit(x, y, batch_size=bsize, nb_epoch=train_epochs, verbose=0)
		pf("/model.fit()")
		total_train += bsize*train_epochs
		if not (total_train % 1):
			pf("Total trainings:", total_train)
			pf("Predicting:")
			prediction = model.predict(x)
			pf("/Predicting:")
			pf("Displaying input image [0]")
			view_img("Ideal (Output)", y[0][0])
			view_img("Bent (Input)", x[0][0])
			pf("Displaying prediction image [0][0]")
			pf(prediction[0][0])
			pf("Pred min: ", prediction[0][0].min())
			pf("Pred max: ", prediction[0][0].max())
			#plt.hist(prediction[0][0], bins=256, range=(0.0, 1.0))
			view_img("Train pred ("+str(total_train)+")", prediction[0][0])
			plt.show()
			#exit(0)
			pf("/Displaying prediction image")
	exit(0)

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
				y = (y/255.0) - .5
				#pf("Ideal image regularized:")
				#pf(y)
				#exit(0)
				#pf(y.shape)
				#view_img(img)
				#pf("Output Image")
				y = y.reshape((1,) + y.shape)  # Numpy array with shape (1, 1, 150, 150)
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

datagen_input, datagen_output = init()
load_imgnames()
model = create_nn()
train_bundles(model)
model.save_weights(weight_store)
sys.exit(0)

#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
pf("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)

# round predictions
rounded = [round(x) for x in predictions]
pf(rounded)

# vim:ts=4 ai
