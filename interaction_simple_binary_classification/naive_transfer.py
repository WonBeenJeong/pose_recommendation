import numpy as np
import sys
import os
import math

import caffe

from sklearn.svm import SVC, SVR
from sklearn import linear_model
from sklearn.decomposition import PCA

def get_accu(pred, tru):
	len_ = len(pred)
	res = 0
	for id in range(len_):
		#if pred[id] == tru[id]:
		#	res += 1
		if pred[id] >= 0.5 and tru[id] == 1:
			res += 1
		if pred[id] < 0.5 and tru[id] == 0:
			res += 1

	return float(res) * 1.0 / len_

def get_features_and_labels(f, flag):
	'''
first time, write the features to a file for future use
	model_file = "./models/bvlc_alexnet.caffemodel"
	model_define="./models/alexnet_deploy.prototxt"
	mean_file = "/home/tianli/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy"
	net = caffe.Net(model_define, model_file, caffe.TEST)

	layer = 'fc7' # extract the features here
    # the n-1 layer = 'fc7'

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
	transformer.set_transpose('data',(2,0,1))
	transformer.set_raw_scale('data', 255.0)

	net.blobs['data'].reshape(1,3,227,227)

	features=[]
	labels =[]
	#fout = None
	#if not os.path.isfile("features.txt"):
	print 'now extracting features from layer ' + layer
	if flag == 1:
		fout = open("train_features.txt",'w')
	else:
		fout = open("test_features.txt",'w')
	for line in open(f, "r"):
		line = line.strip("\n").strip().split(" ")
		img_ = caffe.io.load_image(str(line[0])) # ok

		net.blobs['data'].data[...] = transformer.preprocess('data', img_) # (3, 227, 227) shape is correct
		output = net.forward()
    	#print("inference for image {} done".format(line[0]))
		print(len(net.blobs[layer].data[0])) # 4096

		for item in net.blobs[layer].data[0]:
		    fout.write(str(item)+" ")
		fout.write("\n")
		features.append(net.blobs[layer].data[0])
        #features.append(flatten_again)
		labels.append(float(line[1]))
	fout.close()
	
	return np.asarray(features), np.asarray(labels)

	'''
	features=[]
	labels=[]
	exclude=[]

	id_ = 0
	for line in open(f,'r'):
		line = line.strip("\n").strip().split(" ")
		if flag == 1 and float(line[1]) == 0 and len(exclude) < 100:
			exclude.append(id_)
			id_ += 1
			continue
		id_ += 1
		labels.append(float(line[1]))
	
	if flag == 1:
		file_ = "train_features.txt"
	else:
		file_ = "test_features.txt"

	id_ = 0
	for line in open(file_,'r'):
		line = line.strip("\n").strip().split(" ")
		if flag == 1 and id_ not in exclude:
			features.append([float(item) for item in line])
		if flag == 0:
			features.append([float(item) for item in line])
		id_ += 1
	return np.asarray(features), np.asarray(labels)


def main():
	train_features, train_labels = get_features_and_labels("train.txt", 1)
	test_features, test_labels = get_features_and_labels("test.txt", 0)

	'''
	Binary classification
	classifier = SVC(C = 0.8)
	classifier.fit(train_features, train_labels)

	pred_test = classifier.predict(train_features)
	pred = classifier.predict(test_features)

	'''

	'''
	Do regression

	'''

	clf = linear_model.Lasso(alpha = 1)
	pca = PCA(n_components=50)
	
	train_features_reduced = pca.fit_transform(train_features)
	test_features_reduced = pca.fit_transform(test_features)

	
	clf.fit(train_features_reduced, train_labels)
	pred = clf.predict(test_features_reduced)
	pred_train = clf.predict(train_features_reduced)

	test_accu = get_accu(pred, test_labels)
	train_accu = get_accu(pred_train, train_labels)

	for i in range(len(train_labels)):
		print("real label from training: {}, prediction: {}".format(train_labels[i], pred_train[i]))

	for i in range(len(test_labels)):
		print("real label from testing: {}, prediction: {}".format(test_labels[i], pred[i]))

	print("training accuracy is {}, and testing accuracy is {}.".format(train_accu, test_accu))

if __name__ == '__main__':
	main()