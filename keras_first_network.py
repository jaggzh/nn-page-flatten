#!/usr/bin/python
from __future__ import print_function # For eprint
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import sys
from os import listdir
from os.path import isdir, isfile, join

#from keras.datasets import cifar10
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#print(X_train)
#sys.exit(0)

bent_dir = "blend/pages"
ideal_dir = "blend/ideal"
imgids=[]
verbose=0
datagen=None
img_width=67
img_height=67

## Functions
def exit(ec):
	sys.exit(0)
def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)
def vprint(verbosity, *args, **kwargs):
	if (verbose >= verbosity):
		print(*args, **kwargs)
def imgs_add(name):
	if not isdir(join(ideal_dir, name)):
		eprint("Dir " + name + " does not exist in ideal folder: " + ideal_dir)
	imgids.append(name)
def load_imgnames():
	vprint(2, "Opening " + bent_dir + "\n")
	for d in listdir(bent_dir):
		ifile=join(bent_dir, d)
		if isdir(ifile):
			imgs_add(d)
def init():
	# fix random seed for reproducibility
	seed = 7
	np.random.seed(seed)
	datagen = ImageDataGenerator(
		rotation_range=1,
		width_shift_range=0.1,
		height_shift_range=0.1,
		shear_range=0.1,
		zoom_range=0.1,
		horizontal_flip=False,
		fill_mode='nearest')
	return datagen
def create_nn():
	model = Sequential()
	model.add(
		Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, img_width, img_height)))
	#model.add(Flatten())
	#model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
	#model.add(Dense(8, init='uniform', activation='relu'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
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
def train_nn(model):
	for imgid in imgids:
		ideal_imgs = imgset_ideal(imgid)
		bent_imgs = imgset_bent(imgid)
		for ideal in ideal_imgs:
			img = load_img(ideal)  # PIL image
			y = img_to_array(img)  # Numpy array with shape (1, 150, 150)
			print("Output Image")
			y = y.reshape((1,) + y.shape)  # Numpy array with shape (1, 1, 150, 150)
			print("Output Image")
			print(y)
			for bent in bent_imgs:
				print("Training bent->ideal\n     " + bent + "\n  -> " + ideal)
				img = load_img(bent)  # PIL image
				x = img_to_array(img)  # Numpy array with shape (1, 150, 150)
				x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 3, 150, 150)
				i = 0
				for batch in datagen.flow(x, batch_size=10):
					print(batch)
					print("-- Fitting ----------------\n")
					history = model.fit(batch, y, batch_size=2, nb_epoch=2, verbose=1)
					print("-- Fitting History --------\n")
					print(history.history)
					i += 1
					if i > 20:
						break  # otherwise the generator would loop indefinitely

datagen = init()
load_imgnames()
model = create_nn()
train_nn(model)
sys.exit(0)

#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

pile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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
