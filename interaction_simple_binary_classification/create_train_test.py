import random
import sys
import os

fout = open("all_label.txt",'w')

files=["beach_label.txt", "forest_person_label.txt", "all_background_label.txt"]


for line in open("beach_label.txt",'r'):
	line = line.strip("\n").strip().split(" ")
	a = "/home/tianli/face_detection/results_one_person/original/google_beach_person/" + str(line[0])+".jpg"
	print line
	fout.write(str(a)+" "+str(line[1])+"\n")

for line in open("forest_person_label.txt",'r'):
	line = line.strip("\n").strip().split(" ")
	a = "/home/tianli/face_detection/results_one_person/original/forest_person/" + str(line[0])+".jpg"
	fout.write(str(a)+" "+str(line[1])+"\n")

for line in open("all_background_label.txt",'r'):
	line = line.strip("\n").strip().split(" ")
	a = "/home/tianli/face_detection/results_one_person/original/all_background/" + str(line[0])+".jpg"
	fout.write(str(a)+" "+str(line[1])+"\n")
fout.close()

imgs = []

for line in open("all_label.txt",'r'):
	imgs.append(line)

random.shuffle(imgs, random.random)

fout1 = open("train.txt",'w')
fout2 = open("test.txt",'w')

for img in imgs[:250]:
	fout1.write(str(img))
fout1.close()

for img in imgs[250:]:
	fout2.write(str(img))
fout2.close()