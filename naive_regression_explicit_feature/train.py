'''
	- Subsample from all backgrounds, feature: background (image segmentation, or others) + explicit pose (from any pose estimation model) + (position??)
	- Train a regression model
	- Test on other images
'''

import os
import numpy as np
import cv2
import sys

from sklearn.svm import SVR
from sklearn import linear_model
from sklearn import grid_search
from sklearn.decomposition import PCA

from read_data import *

def get_loss(pred, tru):
    return ((pred - tru) ** 2).mean(axis=None)



def main():
	_, train_feature, train_label = extract_feature_and_label(1)
	print(str(len(train_feature)) + " " + str(len(train_label)))
	test_img_name, test_feature, test_label = extract_feature_and_label(0)

	clf = linear_model.Lasso(alpha = float(sys.argv[5]))
	
	#pca = PCA(n_components=int(sys.argv[3]))
	#train_feature_reduced = pca.fit_transform(train_feature)
	#test_feature_reduced = pca.fit_transform(test_feature)
	
	#clf.fit(train_feature_reduced, train_label)
	
	clf.fit(train_feature, train_label)
	# print prediction on training
	id = 0
	for res in clf.predict(train_feature):
		print(str(train_label[id])+" "+str(float(res)))
		id += 1
	# print prediction on testing
	pred = clf.predict(test_feature)
	pred_train = clf.predict(train_feature)

	id = 0
	for res in pred:
		print(str(test_img_name[id]) + " " + str(test_label[id])+" "+str(float(res)))
		id += 1

    
	loss_train = get_loss(pred_train, train_label)
	loss_test = get_loss(pred, test_label)

	print('mse loss for testing is: ' + str(loss_test) + ", for training is: " + str(loss_train))

if __name__ == '__main__':
	main()