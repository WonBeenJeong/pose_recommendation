from sklearn.cluster import SpectralClustering
import cmath
import numpy as np
import os
import sys
from sklearn import metrics



def get_features(output_dir):

    fin = open(output_dir, 'r')

    features=[]
    name=[]

    for img in os.listdir("/home/tianli/face_detection/results_one_person/original/google_beach_person"):
        name.append(str(img))

    for line in fin:
        line = line.strip("\n").strip().split(" ")
        tmp=[]
        for i in line:
            tmp.append(float(i))
        features.append(tmp)
    fin.close()
    return name, features
'''
try:
	os.system("rm -r result3")
except:
	None
os.system("mkdir result3")
os.system("mkdir result3/0 result3/1 result3/2")
'''

imgs, features = get_features("feature_pool2_train.txt")

for index, gamma in enumerate((0.001,0.01,0.1)):
    for index, k in enumerate(( 2,3,4,5,6, 7, 8, 9, 10, 11, 12)):
        y_pred = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(np.array(features))
        print "Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", k,"score:", metrics.calinski_harabaz_score(np.array(features), y_pred)

spec = SpectralClustering(n_clusters=5, gamma=0.01).fit(np.array(features))

id = 0

for label in spec.labels_:
        os.system("cp /home/tianli/face_detection/results_one_person/original/google_beach_person/" + str(imgs[id]) + " result3/" + str(label))
        id = id + 1
