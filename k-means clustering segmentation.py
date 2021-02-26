# Clustering is the process of grouping objects in such a way that objects in a group are more similar to one
# another than those from other groups. One application of clustering is in the automatic segmentation of regions
# of interest in images. For this machine problem, you will apply kmeans clustering to isolate parasites in blood
# smear images.


import sys
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp
from sklearn.cluster import KMeans

def plasmodium_random():

	img1 = cv2.imread('plasmodium/11c.jpg', -1)
	img2 = cv2.imread('plasmodium/7c.jpg', -1)
	# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
	# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

	img1 = cv2.resize(img1, (0,0), fx = 0.25, fy= 0.25)
	img2 = cv2.resize(img2, (0,0), fx = 0.25, fy= 0.25)
	newImg1 = img1.reshape((-1,3))
	newImg2 = img2.reshape((-1,3))
	concatImg = np.concatenate((newImg1, newImg2))


	f = cv2.imread('plasmodium/1c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output1c.jpg',f)

	f = cv2.imread('plasmodium/3c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output3c.jpg',f)

	f = cv2.imread('plasmodium/6c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output6c.jpg',f)

	f = cv2.imread('plasmodium/7c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output7c.jpg',f)

	f = cv2.imread('plasmodium/11c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output11c.jpg',f)

	f = cv2.imread('plasmodium/19c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output19c.jpg',f)

	f = cv2.imread('plasmodium/55c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output55c.jpg',f)

	f = cv2.imread('plasmodium/79c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output79c.jpg',f)

	f = cv2.imread('plasmodium/94c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output94c.jpg',f)

	f = cv2.imread('plasmodium/105c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output105c.jpg',f)


def filaria_random():

	img1 = cv2.imread('filarioidea/filaria9.jpg', -1)
	img2 = cv2.imread('filarioidea/filaria10.jpg', -1)
	### WAIT RESIZE MUNA PUTA
	img1 = cv2.resize(img1, (0,0), fx = 0.25, fy= 0.25)
	img2 = cv2.resize(img2, (0,0), fx = 0.25, fy= 0.25)
	newImg1 = img1.reshape((-1,3))
	newImg2 = img2.reshape((-1,3))
	concatImg = np.concatenate((newImg1, newImg2))

	directory = "filarioidea/"
	nlist = ["", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

	for filename in nlist :

		f = cv2.imread("filarioidea/filaria" + filename + ".jpg" , -1)
		f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	
		new_f = f.reshape((-1,3))
		kmeans = KMeans(n_clusters=3, random_state=0).fit(concatImg)

		pixels = [0,0,0]

		f_array = kmeans.predict(new_f)
		for i in f_array:
			pixels[i] += 1

		f_array = f_array.reshape((f.shape[0], -1))

		# print pixels

		for i in range(f.shape[0]):
			for j in range(f.shape[1]):
				if f_array[i][j] != 2:
					f[i][j] = [0,0,0]

		cv2.imwrite('filaria-output' + filename + '.jpg',f)

def schistosoma_random():
	img1 = cv2.imread('schistosoma/schistosoma6.jpg', -1)
	img2 = cv2.imread('schistosoma/schistosoma7.jpg', -1)
	### WAIT RESIZE MUNA PUTA
	img1 = cv2.resize(img1, (0,0), fx = 0.25, fy= 0.25)
	img2 = cv2.resize(img2, (0,0), fx = 0.25, fy= 0.25)
	newImg1 = img1.reshape((-1,3))
	newImg2 = img2.reshape((-1,3))
	concatImg = np.concatenate((newImg1, newImg2))

	directory = "schistosoma/"
	nlist = ["", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

	for filename in nlist :

		f = cv2.imread("schistosoma/schistosoma" + filename + ".jpg" , -1)
		f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	
		new_f = f.reshape((-1,3))
		kmeans = KMeans(n_clusters=3, random_state=0).fit(concatImg)

		pixels = [0,0,0]

		f_array = kmeans.predict(new_f)
		for i in f_array:
			pixels[i] += 1

		f_array = f_array.reshape((f.shape[0], -1))

		print pixels

		for i in range(f.shape[0]):
			for j in range(f.shape[1]):
				if f_array[i][j] != 2:
					f[i][j] = [0,0,0]

		cv2.imwrite('schistosoma-output' + filename + '.jpg',f)

def filaria():

	img1 = cv2.imread('filarioidea/filaria9.jpg', -1)
	img2 = cv2.imread('filarioidea/filaria10.jpg', -1)
	img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
	### WAIT RESIZE MUNA PUTA
	img1 = cv2.resize(img1, (0,0), fx = 0.25, fy= 0.25)
	img2 = cv2.resize(img2, (0,0), fx = 0.25, fy= 0.25)
	newImg1 = img1.reshape((-1,3))
	newImg2 = img2.reshape((-1,3))
	concatImg = np.concatenate((newImg1, newImg2))

	directory = "filarioidea/"
	nlist = ["", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

	for filename in nlist :

		f = cv2.imread("filarioidea/filaria" + filename + ".jpg" , -1)
		f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
		f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	
		new_f = f.reshape((-1,3))
		kmeans = KMeans(n_clusters=3, random_state=0).fit(concatImg)

		pixels = [0,0,0]

		f_array = kmeans.predict(new_f)
		for i in f_array:
			pixels[i] += 1

		f_array = f_array.reshape((f.shape[0], -1))

		# print pixels

		for i in range(f.shape[0]):
			for j in range(f.shape[1]):
				if f_array[i][j] != 2:
					f[i][j] = [0,0,0]

		cv2.imwrite('filaria-output' + filename + '.jpg',f)

def schistosoma():
	img1 = cv2.imread('schistosoma/schistosoma6.jpg', -1)
	img2 = cv2.imread('schistosoma/schistosoma7.jpg', -1)
	img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
	### WAIT RESIZE MUNA PUTA
	img1 = cv2.resize(img1, (0,0), fx = 0.25, fy= 0.25)
	img2 = cv2.resize(img2, (0,0), fx = 0.25, fy= 0.25)
	newImg1 = img1.reshape((-1,3))
	newImg2 = img2.reshape((-1,3))
	concatImg = np.concatenate((newImg1, newImg2))

	directory = "schistosoma/"
	nlist = ["", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

	for filename in nlist :

		f = cv2.imread("schistosoma/schistosoma" + filename + ".jpg" , -1)
		f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
		f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	
		new_f = f.reshape((-1,3))
		kmeans = KMeans(n_clusters=3, random_state=0).fit(concatImg)

		pixels = [0,0,0]

		f_array = kmeans.predict(new_f)
		for i in f_array:
			pixels[i] += 1

		f_array = f_array.reshape((f.shape[0], -1))

		print pixels

		for i in range(f.shape[0]):
			for j in range(f.shape[1]):
				if f_array[i][j] != 2:
					f[i][j] = [0,0,0]

		cv2.imwrite('schistosoma-output' + filename + '.jpg',f)

def plasmodium_morethan2():

	img1 = cv2.imread('plasmodium/11c.jpg', -1)
	img2 = cv2.imread('plasmodium/7c.jpg', -1)

	img1 = cv2.resize(img1, (0,0), fx = 0.25, fy= 0.25)
	img2 = cv2.resize(img2, (0,0), fx = 0.25, fy= 0.25)
	newImg1 = img1.reshape((-1,3))
	newImg2 = img2.reshape((-1,3))
	concatImg = np.concatenate((newImg1, newImg2))


	f = cv2.imread('plasmodium/1c.jpg', -1)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=4, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	# for i in f_array:
	# 	pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output1c.jpg',f)

	f = cv2.imread('plasmodium/3c.jpg', -1)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=4, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	# for i in f_array:
	# 	pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output3c.jpg',f)

	f = cv2.imread('plasmodium/6c.jpg', -1)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=4, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	# for i in f_array:
	# 	pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output6c.jpg',f)

	f = cv2.imread('plasmodium/7c.jpg', -1)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=4, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	# for i in f_array:
	# 	pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output7c.jpg',f)

	f = cv2.imread('plasmodium/11c.jpg', -1)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=4, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	# for i in f_array:
	# 	pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output11c.jpg',f)

	f = cv2.imread('plasmodium/19c.jpg', -1)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=4, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	# for i in f_array:
	# 	pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output19c.jpg',f)

	f = cv2.imread('plasmodium/55c.jpg', -1)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=4, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	# for i in f_array:
	# 	pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output55c.jpg',f)

	f = cv2.imread('plasmodium/79c.jpg', -1)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=4, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	# for i in f_array:
	# 	pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output79c.jpg',f)

	f = cv2.imread('plasmodium/94c.jpg', -1)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=4, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	# for i in f_array:
	# 	pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output94c.jpg',f)

	f = cv2.imread('plasmodium/105c.jpg', -1)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=4, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	# for i in f_array:
	# 	pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output105c.jpg',f)


def filaria_morethan2():

	img1 = cv2.imread('filarioidea/filaria9.jpg', -1)
	img2 = cv2.imread('filarioidea/filaria10.jpg', -1)
	### WAIT RESIZE MUNA PUTA
	img1 = cv2.resize(img1, (0,0), fx = 0.25, fy= 0.25)
	img2 = cv2.resize(img2, (0,0), fx = 0.25, fy= 0.25)
	newImg1 = img1.reshape((-1,3))
	newImg2 = img2.reshape((-1,3))
	concatImg = np.concatenate((newImg1, newImg2))

	directory = "filarioidea/"
	nlist = ["", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

	for filename in nlist :

		f = cv2.imread("filarioidea/filaria" + filename + ".jpg" , -1)
		f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	
		new_f = f.reshape((-1,3))
		kmeans = KMeans(n_clusters=4, random_state=0).fit(concatImg)

		pixels = [0,0,0]

		f_array = kmeans.predict(new_f)
		# for i in f_array:
		# 	pixels[i] += 1

		f_array = f_array.reshape((f.shape[0], -1))

		# print pixels

		for i in range(f.shape[0]):
			for j in range(f.shape[1]):
				if f_array[i][j] != 2:
					f[i][j] = [0,0,0]

		cv2.imwrite('filaria-output' + filename + '.jpg',f)

def schistosoma_morethan2():
	img1 = cv2.imread('schistosoma/schistosoma6.jpg', -1)
	img2 = cv2.imread('schistosoma/schistosoma7.jpg', -1)
	### WAIT RESIZE MUNA PUTA
	img1 = cv2.resize(img1, (0,0), fx = 0.25, fy= 0.25)
	img2 = cv2.resize(img2, (0,0), fx = 0.25, fy= 0.25)
	newImg1 = img1.reshape((-1,3))
	newImg2 = img2.reshape((-1,3))
	concatImg = np.concatenate((newImg1, newImg2))

	directory = "schistosoma/"
	nlist = ["", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

	for filename in nlist :

		f = cv2.imread("schistosoma/schistosoma" + filename + ".jpg" , -1)
		f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	
		new_f = f.reshape((-1,3))
		kmeans = KMeans(n_clusters=4, random_state=0).fit(concatImg)

		pixels = [0,0,0]

		f_array = kmeans.predict(new_f)
		# for i in f_array:
		# 	pixels[i] += 1

		f_array = f_array.reshape((f.shape[0], -1))

		# print pixels

		for i in range(f.shape[0]):
			for j in range(f.shape[1]):
				if f_array[i][j] != 2:
					f[i][j] = [0,0,0]

		cv2.imwrite('schistosoma-output' + filename + '.jpg',f)

def plasmodium_pixel():

	img1 = cv2.imread('plasmodium/11c.jpg', -1)
	img2 = cv2.imread('plasmodium/7c.jpg', -1)
	# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
	# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
	pixelArray = np.array([(614,472,0),(424,588,0)])
	img1 = cv2.resize(img1, (0,0), fx = 0.25, fy= 0.25)
	img2 = cv2.resize(img2, (0,0), fx = 0.25, fy= 0.25)
	newImg1 = img1.reshape((-1,3))
	newImg2 = img2.reshape((-1,3))
	concatImg = np.concatenate((newImg1, newImg2))


	f = cv2.imread('plasmodium/1c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, init=pixelArray, n_init=1, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output1c.jpg',f)

	f = cv2.imread('plasmodium/3c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, init=pixelArray, n_init=1, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output3c.jpg',f)

	f = cv2.imread('plasmodium/6c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, init=pixelArray, n_init=1, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output6c.jpg',f)

	f = cv2.imread('plasmodium/7c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, init=pixelArray, n_init=1, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output7c.jpg',f)

	f = cv2.imread('plasmodium/11c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, init=pixelArray, n_init=1, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output11c.jpg',f)

	f = cv2.imread('plasmodium/19c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, init=pixelArray, n_init=1, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output19c.jpg',f)

	f = cv2.imread('plasmodium/55c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, init=pixelArray, n_init=1, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output55c.jpg',f)

	f = cv2.imread('plasmodium/79c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, init=pixelArray, n_init=1, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output79c.jpg',f)

	f = cv2.imread('plasmodium/94c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, init=pixelArray, n_init=1, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output94c.jpg',f)

	f = cv2.imread('plasmodium/105c.jpg', -1)
	# f = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
	f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	new_f = f.reshape((-1,3))
	kmeans = KMeans(n_clusters=2, init=pixelArray, n_init=1, random_state=0).fit(concatImg)
	pixels = [0,0,0]
	f_array = kmeans.predict(new_f)
	for i in f_array:
		pixels[i] += 1

	f_array = f_array.reshape((f.shape[0], -1))

	# print pixels

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f_array[i][j] != 1:
				f[i][j] = [0,0,0]
	cv2.imwrite('plasmodium-output105c.jpg',f)


def filaria_pixel():

	img1 = cv2.imread('filarioidea/filaria9.jpg', -1)
	img2 = cv2.imread('filarioidea/filaria10.jpg', -1)
	### WAIT RESIZE MUNA PUTA
	
	pixelArray = np.array([(485,257,0),(500,360,0),(600,600,0)])

	img1 = cv2.resize(img1, (0,0), fx = 0.25, fy= 0.25)
	img2 = cv2.resize(img2, (0,0), fx = 0.25, fy= 0.25)
	newImg1 = img1.reshape((-1,3))
	newImg2 = img2.reshape((-1,3))
	concatImg = np.concatenate((newImg1, newImg2))

	directory = "filarioidea/"
	nlist = ["", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

	for filename in nlist :

		f = cv2.imread("filarioidea/filaria" + filename + ".jpg" , -1)
		f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	
		new_f = f.reshape((-1,3))
		kmeans = KMeans(n_clusters=3, init=pixelArray, n_init=1, random_state=0).fit(concatImg)

		pixels = [0,0,0]

		f_array = kmeans.predict(new_f)
		for i in f_array:
			pixels[i] += 1

		f_array = f_array.reshape((f.shape[0], -1))

		# print pixels

		for i in range(f.shape[0]):
			for j in range(f.shape[1]):
				if f_array[i][j] != 2:
					f[i][j] = [0,0,0]

		cv2.imwrite('filaria-output' + filename + '.jpg',f)

def schistosoma_pixel():
	img1 = cv2.imread('schistosoma/schistosoma6.jpg', -1)
	img2 = cv2.imread('schistosoma/schistosoma7.jpg', -1)
	### WAIT RESIZE MUNA PUTA

	pixelArray = np.array([(340,379,0),(482,233,0),(772,70,0)])
	img1 = cv2.resize(img1, (0,0), fx = 0.25, fy= 0.25)
	img2 = cv2.resize(img2, (0,0), fx = 0.25, fy= 0.25)
	newImg1 = img1.reshape((-1,3))
	newImg2 = img2.reshape((-1,3))
	concatImg = np.concatenate((newImg1, newImg2))

	directory = "schistosoma/"
	nlist = ["", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

	for filename in nlist :

		f = cv2.imread("schistosoma/schistosoma" + filename + ".jpg" , -1)
		f = cv2.resize(f, (0,0), fx = 0.25, fy= 0.25)
	
		new_f = f.reshape((-1,3))
		kmeans = KMeans(n_clusters=3, init=pixelArray, n_init=1, random_state=0).fit(concatImg)

		pixels = [0,0,0]

		f_array = kmeans.predict(new_f)
		for i in f_array:
			pixels[i] += 1

		f_array = f_array.reshape((f.shape[0], -1))

		# print pixels

		for i in range(f.shape[0]):
			for j in range(f.shape[1]):
				if f_array[i][j] != 2:
					f[i][j] = [0,0,0]

		cv2.imwrite('schistosoma-output' + filename + '.jpg',f)

def main():
	choice1 = input("Choose a data set: \n1. Plasmodium \n2. Filaria \n3. Schistosoma\n")

	if(choice1 == 1):
		choice = input("Segmentation: \n1. 2 Random Centroids\n2. Pixel Locations\n3. More than 2 centroids\n4. Different Color Space\n") 

		if choice == 1:
			plasmodium_random()
		elif choice == 2:
			plasmodium_pixel()
		elif choice == 3:
			plasmodium_morethan2()
		else:
			plasmodium_random()

	elif(choice1 == 2):
		choice = input("Segmentation: \n1. 2 Random Centroids\n2. Pixel Locations\n3. More than 2 centroids\n4. Different Color Space\n") 

		if choice == 1:
			filaria_random()
		elif choice == 2:
			filaria_pixel()
		elif choice == 3:
			filaria_morethan2()
		else:
			filaria()

	else:
		choice = input("Segmentation: \n1. 2 Random Centroids\n2. Pixel Locations\n3. More than 2 centroids\n4. Different Color Space\n") 

		if choice == 1:
			schistosoma_random()
		elif choice == 2:
			schistosoma_pixel()
		elif choice == 3:
			schistosoma_morethan2()
		else:
			schistosoma()

main()