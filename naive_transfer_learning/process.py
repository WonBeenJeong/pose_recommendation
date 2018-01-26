import os
import sys

fin = open("test_label.txt",'r')
fout = open('test_label2.txt','w')
for line in fin:
	line = line.strip().split(" ")
	fout.write("/home/tianli/face_detection/results_one_person/original/forest_person/"+str(line[0])+".jpg"+" "+str(line[1])+"\n")
