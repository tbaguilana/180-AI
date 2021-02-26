import sys
import os
import cv2
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.decomposition import PCA 

def ANN():
	training_set = []
	train_labels = []
	test_set = []
	test_labels = []
	for x in range(1, 41):
		for y in range(1,7):
			string = "./att_faces/s" + str(x) + "/"
			labelstring = "./att_faces/s" + str(x)
			string = string + str(y) + ".pgm"
			img = cv2.imread(string, -1)
			img = cv2.resize(img, (0,0), fx = 0.5, fy= 0.5)
			# img = img.reshape(-1)
			# print img.size
			new_img = cv2.equalizeHist(img).flatten()
			training_set.append(new_img)
			train_labels.append(labelstring)

		for y in range(7,11):
			string = "./att_faces/s" + str(x) + "/"
			labelstring = "./att_faces/s" + str(x)
			string = string + str(y) + ".pgm"
			img = cv2.imread(string, -1)
			img = cv2.resize(img, (0,0), fx = 0.5, fy= 0.5)
			# img = img.reshape(-1)
			# print img.size
			new_img = cv2.equalizeHist(img).flatten()
			test_set.append(new_img)
			test_labels.append(labelstring)

	training_set = np.array(training_set)
	train_labels = np.array(train_labels)
	test_set = np.array(test_set)
	test_labels = np.array(test_labels)

	scaler = StandardScaler()
	scaler.fit(training_set)
	scaled_set = scaler.transform(training_set)
	scaled_set = np.array(scaled_set)

	scaled_test = scaler.transform(test_set)

	# 3. 
	# pca = PCA(n_components = 10)
	# pca.fit(scaled_set)
	# scaled_set = pca.transform(scaled_set)
	# scaled_test = pca.transform(scaled_test)

	# scaler.fit(test_set, test_labels)

	mlp = MLPClassifier(activation = 'logistic', hidden_layer_sizes=(1288))
	mlp.fit(scaled_set, train_labels)

	print("Training Set:") 
	print(mlp.score(scaled_set, train_labels)*100)
	print("Test Set:  " ) 
	print(mlp.score(scaled_test, test_labels)*100)

def SVM():
	training_set = []
	train_labels = []
	test_set = []
	test_labels = []
	for x in range(1, 41):
		for y in range(1,7):
			string = "./att_faces/s" + str(x) + "/"
			labelstring = "./att_faces/s" + str(x)
			string = string + str(y) + ".pgm"
			img = cv2.imread(string, -1)
			img = cv2.resize(img, (0,0), fx = 0.5, fy= 0.5)
			# img = img.reshape(-1)
			# print img.size
			new_img = cv2.equalizeHist(img).flatten()
			training_set.append(new_img)
			train_labels.append(labelstring)

		for y in range(7,11):
			string = "./att_faces/s" + str(x) + "/"
			labelstring = "./att_faces/s" + str(x)
			string = string + str(y) + ".pgm"
			img = cv2.imread(string, -1)
			img = cv2.resize(img, (0,0), fx = 0.5, fy= 0.5)
			# img = img.reshape(-1)
			# print img.size
			new_img = cv2.equalizeHist(img).flatten()
			test_set.append(new_img)
			test_labels.append(labelstring)

	training_set = np.array(training_set)
	train_labels = np.array(train_labels)
	test_set = np.array(test_set)
	test_labels = np.array(test_labels)

	scaler = StandardScaler()
	scaler.fit(training_set)
	scaled_set = scaler.transform(training_set)
	scaled_set = np.array(scaled_set)

	scaled_test = scaler.transform(test_set)

	# scaler.fit(test_set, test_labels)
	clf = svm.SVC()
	# clf = svm.SVC(kernel = 'poly', degree = 5) #5
	# clf = svm.SVC(kernel = 'rbf', gamma = 1) #6

	clf.fit(scaled_set, train_labels)

	print("Training Set: ")
	print((clf.score(scaled_set, train_labels))*100)
	print("Test Set: ")
	print((clf.score(scaled_test, test_labels))*100)

def main():

	print("Enter Choice:\n0. ANN\n1. SVM\n")
	x = input()
	if(x == 0):
		ANN()
	else: 
		SVM()

main()
