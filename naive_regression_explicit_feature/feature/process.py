import os

fin = open("coordinate_forest.txt",'r')
fout = open("coordinate_forest2.txt",'w')

fin2 = open("coordinate_beach.txt",'r')
fout2 = open("coordinate_beach2.txt",'w')

for line in fin:
	line = line.strip("\n").split(" ")
	line[0] = "/home/tianli/face_detection/results_one_person/original/forest_person/" + str(line[0])
	for i in range(0, len(line)):
		fout.write(str(line[i]) + " ")
	fout.write("\n")
for line in fin2:
	line = line.strip("\n").split(" ")
        line[0] = "/home/tianli/face_detection/results_one_person/original/google_beach_person/" + str(line[0])
        for i in range(0, len(line)):
                fout2.write(str(line[i]) + " ")
	fout2.write("\n")
