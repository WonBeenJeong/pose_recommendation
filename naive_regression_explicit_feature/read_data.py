import numpy as np
import os
import sys
from utils import *

POSE_DIR='/home/tianli/pose_estimation/pose-tensorflow/'



def model_pose(values):
		raw_dis = []
		num = len(values)
		max_ = 0.0
		
		head_x = float(values[num-2])
		head_y = float(values[num-1])

        # feature of distances
		for i in range(0,num, 2):
			for j in range(i + 2, num, 2):
				if i != j:
					temp = (float(values[i]) - float(values[j]))*(float(values[i]) - float(values[j])) + (float(values[i + 1]) - float(values[j + 1])) * (float(values[i+1])-float(values[j+1]))
					temp = temp ** 0.5
					if temp > max_:
						max_ = temp
					raw_dis.append(temp)

		for i in range(len(raw_dis)):
			raw_dis[i] = raw_dis[i] * 1.0 / max_

        #feature of angles
		for i in range(0, num, 2):
			for j in range(0, num, 2):
				if i != j:
					v1 = np.array([float(values[i]) - float(values[j]), float(values[i+1]) - float(values[j+1])])
					v2 = np.array([0, -1])
					angle = angle_between(v1, v2)
                #print(angle)
					raw_dis.append(angle*float(sys.argv[1]))

        #feature of relative position of differnet body parts
		for i in range(0, num-2, 2):
			raw_dis.append((float(values[i]) - float(values[num-1])) / abs(float(values[i]) - float(values[num-1])))


        #feature of relative angles
		(chin_x, chin_y) = (float(values[num-4]), float(values[num-3]))
		for i in range(0, num-4, 2):
				(left_x, left_y) = (0, 0)
				if i == 0:
					(left_x, left_y) = (float(values[i]), float(values[i+1]))
					(right_x, right_y) = (float(values[10]), float(values[11]))
				if i == 2:
					(left_x, left_y) = (float(values[i]), float(values[i+1]))
					(right_x, right_y) = (float(values[8]), float(values[9]))
				if i == 4:
					(left_x, left_y) = (float(values[i]), float(values[i+1]))
					(right_x, right_y) = (float(values[6]), float(values[7]))
				if i == 12:
					(left_x, left_y) = (float(values[i]), float(values[i+1]))
					(right_x, right_y) = (float(values[22]), float(values[23]))
				if i == 14:
					(left_x, left_y) = (float(values[i]), float(values[i+1]))
					(right_x, right_y) = (float(values[20]), float(values[21]))
				if i == 16:
					(left_x, left_y) = (float(values[i]), float(values[i+1]))
					(right_x, right_y) = (float(values[18]), float(values[19 ]))
				if left_x > 0 and left_y > 0:
					v1 = np.array([left_x - chin_x, left_y - chin_y])
					v2 = np.array([right_x - chin_x, right_y - chin_y])
					angle = angle_between(v1, v2)
					raw_dis.append(angle * float(sys.argv[2]))
#
					v1_2 = np.array([left_x - head_x, left_y - head_y])
					v2_2 = np.array([right_x - head_x, right_y - head_y])
					angle_2 = angle_between(v1_2, v2_2)
					raw_dis.append(angle_2 * float(sys.argv[2]))
		return raw_dis

def pose_feature_extraction(coordinates):
	#features=[]
	#for values in coordinates:	
	#cc	features.append(model_pose(values))
	return np.asarray(model_pose(coordinates))

def extract_feature_and_label(train=1):
	coordinates = {}
	labels = []
	scene={}
	composition={}
	features=[]
	image_name=[]
	if train == 1:
		for line in open("data/train_now.txt",'r'):
			line = line.strip("\n").strip().split(" ")
			# a map from image name to its label, next, extract the features for those images
			labels.append(float(line[1]))
			image_name.append(str(line[0]))

	else:
		for line in open("data/test_now.txt",'r'):
			line = line.strip("\n").strip().split(" ")
			labels.append(float(line[1]))
			image_name.append(line[0])

	'''
	 extract key point coordinates for all images
	'''
	for line in open('feature/coordinate_beach.txt','r'):
		line = line.strip("\n").strip().split(" ")
		coordinates[line[0]] = line[1:]
	for line in open('feature/coordinate_forest.txt','r'):
		line = line.strip("\n").strip().split(" ")
		coordinates[line[0]] = line[1:]
	for line in open('feature/coordinate_all_background.txt','r'):
		line = line.strip("\n").strip().split(" ")
		coordinates[line[0]] = line[1:]

	'''
	extract the background scene features for all images
	'''
	for line in open('feature/scene_parsing.txt','r'):
		line = line.strip("\n").strip().split(" ")
		scene[line[0]] = [float(item) * float(sys.argv[3]) for item in line[1:]]
	for line in open('feature/scene_parsing2.txt','r'):
		line = line.strip("\n").strip().split(" ")
		scene[line[0]] = [float(item) * float(sys.argv[3]) for item in line[1:]]

	'''
	extract composition features, multiply it with a scale factor
	'''
	for line in open("feature/composition1.txt",'r'):
		line = line.strip("\n").strip().split(" ")
		composition[line[0]] = [float(item) * float(sys.argv[4]) for item in line[1:]]
	'''
	concatenate the features above, the index is the absolute path of file name
	'''
	for name in image_name:
			
			#one_image_feature=
			#print scene[name]
		a = pose_feature_extraction(coordinates[name])
		b = np.asarray(scene[name])
		c = np.asarray(composition[name])
		one_image_feature = np.concatenate((a, b, c), axis=0)
		features.append(one_image_feature)

	return image_name, np.asarray(features), np.asarray(labels)
	
