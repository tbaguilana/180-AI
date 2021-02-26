# AGUILANA, Trina & HERNANDEZ, Kat
# CS180 PROJECT

import sys
import numpy as np
import cv2
import glob
import random
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def main():
	# lunaPaintings = glob.glob('./luna/*.jpg')
	# monetPaintings = glob.glob('./monet/*.jpg')
	# vanGoghPaintings = glob.glob('./van_gogh/*.jpeg')
	# testPaintings = glob.glob('./test/*.jpeg')
	# testPaintings.reverse()

	print "\nMENU ----------------------------------------"
	directory = raw_input("Choose artist directory: ")
	paintings = glob.glob('./' + directory + '/*.*')
	centroidCount = int(raw_input("Centroid Count: "))
	
	# FILE READING
	print "\nFILE READING --------------------------------"
	for i in range(len(paintings)):		# iterate through all files in directory
		imgData = cv2.imread(paintings[i])
		print "reading "+paintings[i]+"..."
		# !!!!!! if we'll change colorspace, insert here
		imgData = cv2.resize(imgData,(0,0),fx=0.25, fy=0.25)
		imgData = np.reshape(imgData,-1).reshape(imgData.shape[0]*imgData.shape[1],imgData.shape[2])
		
		if i == 0:					# first file in data
			paintingsData = imgData
		else:
			paintingsData = np.vstack((paintingsData,imgData))
	print "... done reading all "+str(len(paintings))+" files!"

	# KMEANS FITTING OF INPUT
	print "\nKMEANS --------------------------------------"
	kmeans = KMeans(n_clusters=centroidCount,random_state=0,max_iter=1000)
	print "Fitting pixel values..."
	kmeans = kmeans.fit(paintingsData)		# fitting of all pixels
	print "Getting cluster labels..."
	labels = kmeans.predict(paintingsData)	# getting cluster labels
	print "Getting most occuring RGB values..."
	colors = kmeans.cluster_centers_		# values of RGB centroids

	print "\nCENTROIDS VALUES ----------------------------"
	for i in range(centroidCount):
		print str(i)+'\t'+str(colors[i][0])+'\t'+str(colors[i][1])+'\t'+str(colors[i][2])

	# SETTING UP THE BAR GRAPH OF THE DOMINANT COLORS
	colorArray = []
	y = []
	for i in range(len(colors)):
		rgb = (colors[i][0]/255, colors[i][1]/255, colors[i][2]/255) #putting values in the [0,1] range of rgb
		colorArray.append(rgb)	#put tuples into an array
		y.append(10)			#height of the graph

	x = range(len(y))			#range of the x axis
	plt.bar(x, y, align = 'center', color=colorArray)	#plot the colors into the bar graph
	plt.axis('off')
	plt.show()

	# PERFORMANCE MEASURE: checks the existence of the RGB values of the centroids
	# across all painting datasets. range is +-20
	# for i in range(len(paintings)):
	# 	imgData = cv2.imread(paintings[i])
	# 	print paintings[i]
	# 	c1,c2,c3 = 0,0,0
		

	# print paintingsData
	# print paintingsData.shape[0]
	# print paintingsData.shape[1]
	# print paintingsData[:,0]
	# print paintingsData[:,1]
	# print paintingsData[:,2]


	fig = plt.figure()
	ax = Axes3D(fig)

	ax.scatter(paintingsData[:,0], paintingsData[:,1], paintingsData[:,2])
	ax.scatter(colors[:, 0], colors[:, 1], colors[:, 2], marker='*', c='#050505', s=1000)
	plt.show()
	
main()
