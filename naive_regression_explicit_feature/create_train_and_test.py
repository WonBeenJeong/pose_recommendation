import os
import sys
import random

DATA_ROOT='/home/tianli/naive_regression_explicit_feature/data/'
DATA_DIR=['forest_person','google_beach_person','all_background']

imgs = []

fin1 = open("data/test.txt",'r')
fin2 = open("data/train.txt",'r')
fin3 = open("data/label2.txt",'r')

for line in fin1:
	imgs.append(line)
for line in fin2:
	imgs.append(line)
for line in fin3:
	imgs.append(line)

random.shuffle(imgs, random.random)

fout1 = open("data/train_now.txt",'w')
fout2 = open("data/test_now.txt",'w')

for img in imgs[:250]:
	fout1.write(str(img))
fout1.close()

for img in imgs[250:]:
	fout2.write(str(img))
fout2.close()

