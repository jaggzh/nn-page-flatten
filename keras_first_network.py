#!/usr/bin/python
from __future__ import print_function # For eprint
from keras.models import Sequential, Model # , load_weights, save_weights
from keras.layers import Dense, Reshape, Flatten, Convolution2D, MaxPooling2D, Input
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import sys
from os import listdir
from os.path import isdir, isfile, join
import matplotlib.pyplot as plt
import random


#from keras.datasets import cifar10
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#print(X_train)
#sys.exit(0)

weight_store = "weights.h5"
bent_dir = "blend/pages"
ideal_dir = "blend/ideal"
img_ids=[]
verbose=0
datagen_input=None
datagen_output=None
img_width=67
img_height=67
max_imagesets=100 # imageset = Each unique page of words (bent or flat)
max_main_imgloops = 100 # Number of times to loop through entire set of images
train_epochs=1
out_batch_versions=3 # number of distorted images to feed in
in_batch_versions=3 # number of distorted images to feed in
load_weights=0      # load prior run stored weights
test_perc = 7

## Functions
def exit(ec):
	sys.exit(0)
def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)
def vprint(verbosity, *args, **kwargs):
	if (verbose >= verbosity):
		print(*args, **kwargs)
def load_imgnames():
	vprint(2, "Opening " + bent_dir + "\n")
	i = 0
	for d in listdir(bent_dir):
		i = i+1
		ifile=join(bent_dir, d)
		if isdir(ifile):        # Valid bent_images dir
			                    # ...corresponding to valid ideal_image dir
			if not isdir(join(ideal_dir, name)):
				eprint("Dir " + name + " not in ideal folder: " + ideal_dir)
			img_ids.append(name)
		if max_imagesets > 0 and i > max_imagesets:
			break
def init():
	# fix random seed for reproducibility
	seed = 7
	np.random.seed(seed)
	datagen_input = ImageDataGenerator(
		rotation_range=1,
		width_shift_range=0.1,
		height_shift_range=0.1,
		shear_range=0.1,
		zoom_range=0.1,
		horizontal_flip=False,
		fill_mode='nearest')
	datagen_output = ImageDataGenerator(
		rotation_range=1,
		width_shift_range=0.1,
		height_shift_range=0.1,
		shear_range=0.1,
		zoom_range=0.1,
		horizontal_flip=False,
		fill_mode='nearest')
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
def create_nn():
	inputs = Input(shape = (3, 67, 67))
	x = Flatten()(inputs)
	x = Dense(100)(x)
	x = Dense(100)(x)
	x = Dense(3*67*67)(x)
	x = Reshape((3, 67, 67))(x)
	#x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(inputs)
	#x = MaxPooling2D((2,2), border_mode='same')(x)

	#x = Dense(1000, input_dim=4, init='uniform', activation='relu')(inputs)
	#x = Dense(1000, input_dim=3, init='uniform', activation='relu')(inputs)
	#model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
	#x = Convolution2D(3, 3, 3, border_mode='same')(x)

	model = Model(input=inputs, output=x)
	model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
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
	return model
def view_img(img):
	plt.figure(figsize=(4,4))
	print(img)
	plt.ion()
	plt.imshow(img)
	plt.draw()
	#exit(0)
def train_nn(model):
	x = None
	y = None
	total_train = 0
	for our_epochs in range(max_main_imgloops):
	for imgid in img_ids:
		ideal_imgs = imgset_ideal(imgid)
		bent_imgs = imgset_bent(imgid)
		for ideal in ideal_imgs:
			print("Training " + ideal)
			img = load_img(ideal)  # PIL image
			y = img_to_array(img)  # Numpy array with shape (1, 150, 150)
			print(y)
			#view_img(img)
			#print("Output Image")
			y = y.reshape((1,) + y.shape)  # Numpy array with shape (1, 1, 150, 150)
			#print("Output Image")
			#print(y)
			j = 0
			for out_batch in datagen_output.flow(y, batch_size=1):
				j += 1
				if j > out_batch_versions:
					break
				for bent in bent_imgs:
					img = load_img(bent)  # PIL image
					x = img_to_array(img)  # Numpy array with shape (1, 150, 150)
					x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 3, 150, 150)
					i = 0
					for in_batch in datagen_input.flow(x, batch_size=2):
						i += 1
						if i > in_batch_versions:
							break  # otherwise the generator would loop indefinitely
						#print(in_batch)
						#print("-- Fitting ----------------\n")
						history = model.fit(in_batch, out_batch, batch_size=2, nb_epoch=train_epochs, verbose=0)
						total_train += 1
						#print("-- Fitting History --------\n")
						#print(history.history)
						if not (total_train % 10):
							print("Trained: {}".format(total_train))
						if not (total_train % 5000):
							prediction = model.predict(x)
							print(prediction[0][0])
							view_img(prediction[0][0])
						#print(prediction)
						#exit(0)
						#scores = model.evaluate(in_batch,out_batch)
						#print("\033[33;1m%s: %.2f%%\033[0m" % (model.metrics_names[1], scores[1]*100))

datagen_input, datagen_output = init()
load_imgnames()
model = create_nn()
train_nn(model)
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
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)

# round predictions
rounded = [round(x) for x in predictions]
print(rounded)

# vim:ts=4 ai
