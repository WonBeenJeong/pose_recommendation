import os

import sys

fin = open("test.txt",'r')
fout = open("test2.txt",'w')

for line in fin:
	line=line.strip("\n").split(" ")
	if os.path.isfile(str(line[0])):
		fout.write(str(line[0])+ " "+str(line[1])+"\n")
