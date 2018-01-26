import sys
import cv2
import numpy as np
import os
from sklearn.svm import SVR
from sklearn import grid_search
from sklearn import linear_model
from sklearn.decomposition import PCA

import caffe

def get_labels(img_dir, label_file, flag):

    score={}
    labels=[]
    for line in open(label_file,'r'):
        line = line.strip().split(" ")
        score[str(line[0])]=float(line[1])
        print "score[" + str(line[0])+"] is:"+str(line[1])

    for img in os.listdir(img_dir):
        labels.append(score[os.path.join(os.path.abspath(img_dir),img)])

    #return labels
    
    if flag == 0:
        return labels[:80]
    else:
        return labels[80:]
    

def get_features(output_dir, flag):

    fin = open(output_dir, 'r')

    features=[]


    for line in fin:
        line = line.strip("\n").strip().split(" ")
        tmp=[]
        for i in line:
            tmp.append(float(i))
        features.append(tmp)
    fin.close()

    #return features
    
    if flag == 0:
        return features[:80]
    else:
        return features[80:]
    

def get_loss(pred, tru):
    return ((pred - tru) ** 2).mean(axis=None)

def main():

    train_features = get_features("feature_fc7_train.txt", 0)
    train_labels = get_labels(sys.argv[1], 'train_label2.txt', 0)
    test_features = get_features("feature_fc7_train.txt", 1)
    test_labels = get_labels(sys.argv[1],'train_label2.txt', 1)
    '''
    parameters = {
    'C':            np.arange( 100, 1000, 100).tolist(),
    'kernel':       ['linear', 'rbf'],                  
    'degree':       np.arange( 0, 10+0, 1 ).tolist(),
    'gamma':        np.logspace( -6, 2, 8).tolist(),
    }

    model = grid_search.RandomizedSearchCV( 
                                        estimator           = SVR(),
                                        param_distributions = parameters,
                                        n_jobs              = 4,
                                        )         # scoring = 'accuracy'
    model.fit(np.asarray(features), np.asarray(labels))
    print( model.best_estimator_ )
    print( model.best_score_ )
    print( model.best_params_ )
    '''
    #'kernel': 'linear', 'C': 300, 'degree': 7, 'gamma': 0.5179474679231202}
    #clf = SVR(C = 300.0, kernel='linear', degree=7, gamma=0.5)
    clf = linear_model.Lasso(alpha = 10)
    pca = PCA(n_components=35)
    features_reduced = pca.fit_transform(np.asarray(train_features))
    test_reduced = pca.fit_transform(np.asarray(test_features))
    clf.fit(np.asarray(features_reduced),np.asarray(train_labels))
    id = 0
    for res in clf.predict(features_reduced):
        print str(train_labels[id])+" "+str(float(res))
        id += 1
    pred = clf.predict(test_reduced)
    for res in pred:
        print str(float(res))
        id += 1

    loss = get_loss(pred, test_labels)
    loss_train = get_loss(clf.predict(features_reduced),train_labels)
    print 'mse loss for testing is: ' + str(loss) + ", for training is: " + str(loss_train)

if __name__ == "__main__":
    main()

# export CUDA_VISIBLE_DEVICES=0 python read2.py /home/tianli/face_detection/results_one_person/original/google_beach_person /home/tianli/face_detection/results_one_person/original/forest_person

# mode =1 : 

