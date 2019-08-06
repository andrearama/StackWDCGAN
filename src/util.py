import os
import numpy as np
import matplotlib.image as mpimg
import cv2
import csv


def getData(path, filenames, size):

	X = []
	# Iterate over train, val and test
	for filename in filenames:

		with open(path + filename, encoding="ISO-8859-1") as csv_file:
			csv_reader = csv.reader(csv_file, delimiter='\t')
			next(csv_reader, None)
			for row in csv_reader:
				if (row[6] == "landscape"):
					img = cv2.resize(mpimg.imread(path + "Images/" + row[0]), (size, size), interpolation = cv2.INTER_CUBIC)
					
					# Check size correctness
					if len(img.shape) != 3 or img.shape[0] != size or img.shape[1] != size or img.shape[2] != 3:
						print("ERROR " + str(img.shape))
						exit()

					# RGB out of range
					if img.max() > 255 or img.min() < 0:
						print("ERROR " + str(path))
						exit()

					# Normalize between [-1, 1]
					img = (img - 127.5) / 127.5
					X.append(img)

	return np.array(X, dtype=np.float32)


def saveImages(filename, images):    
    for i in range(len(images)):
        mpimg.imsave(filename + "-" + str(i) + ".png",  ( (images[i] * 127.5) + 127.5 ).astype(np.uint8) )