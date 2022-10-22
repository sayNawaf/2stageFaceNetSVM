

from os import listdir
import os.path
from numpy import asarray
import face_recognition


from numpy import expand_dims
from numpy import asarray


import pickle


# load images and extract faces for all images in a directory
def load_faces(directory):
    
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
       

        image = face_recognition.load_image_file(path)

        encoding = face_recognition.face_encodings(image)[0]
        # store
        faces.append(encoding)
    return faces



# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not os.path.isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)




def GenDataset(path):
    # load train dataset
    trainX, trainy = load_dataset(path+'/train/')
    
    # load test dataset
    testX, testy = load_dataset(path + '/val/')

    print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
    
    return trainX, trainy, testX, testy
