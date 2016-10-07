#!/usr/bin/python
from __future__ import print_function # For eprint
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import sys
from os import listdir
from os.path import isdir, isfile, join

bent_dir = "blend/pages"
ideal_dir = "blend/ideal"
imgids=[]
verbose=0

## Functions
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
def create_nn():
	model = Sequential()
	model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
	model.add(Dense(8, init='uniform', activation='relu'))
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
		print(bent_imgs)

init()
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
