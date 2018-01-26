import os
import sys

fin = open("label_all_background.txt",'r')
fout = open("label2.txt",'w')

for line in fin:
	line = line.strip("\n").split(" ")
	line[0]="/home/tianli/face_detection/results_one_person/original/all_background/"+str(line[0]+".jpg")
	fout.write(line[0]+" "+str(line[1])+"\n")


