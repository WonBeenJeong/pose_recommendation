import sys
import cv2
import numpy as np
import os
from sklearn.svm import SVR

import caffe

def get_labels(img_dir):

    score={}
    labels=[]
    for line in open('label2.txt','r'):
        line = line.strip().split(" ")
        score[str(line[0])]=float(line[1])

    for img in os.listdir(img_dir):
        labels.append(score[os.path.join(os.path.abspath(img_dir),img)])

    return labels

def get_features(img_dir, output_dir):

    model_file = './bvlc_alexnet.caffemodel'
    model_define = './alexnet_deploy.prototxt'
    mean_file = '/home/tianli/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
    fout = open(output_dir, 'w')

    net = caffe.Net(model_define, model_file, caffe.TEST)

    layer = 'fc7' # extract the features here
    # the n-1 layer = 'fc7'

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
    transformer.set_transpose('data',(2,0,1))
    transformer.set_raw_scale('data', 255.0)

    net.blobs['data'].reshape(1,3,227,227)

    features=[]
    name =[]

    print 'now tracting features from '+layer
    for img in os.listdir(img_dir):
        name.append(str(img))
        img_ = caffe.io.load_image(os.path.join(os.path.abspath(img_dir),img))
        print(img_)
        net.blobs['data'].data[...] = transformer.preprocess('data', img_)
        print(net.blobs['data'].data[...].shape)
        
        print("flag2")
        output = net.forward()
        print len(net.blobs[layer].data[0]) # 4096
        l = net.blobs[layer].data[0]
        #flat_list = [item for sublist in l for item in sublist]
        #flatten_again = [item for sublist in flat_list for item in sublist]
        for item in l:
            fout.write(str(item)+" ")
        fout.write("\n")
        #features.append(net.blobs[layer].data[0])
        features.append(l)
    fout.close()
    return name, features


def main():
    print sys.argv[1]
    print sys.argv[2]
    _, features = get_features(sys.argv[1], "feature_fc7_train.txt")
    #labels = get_labels(sys.argv[1])
    name, test = get_features(sys.argv[2], "feature_fc7_test.txt")
    #clf = SVR(C = 500.0)
    #clf.fit(np.asarray(features), np.asarray(labels))
    #id = 0
    #for res in clf.predict(test):
    #    print str(name[id]), float(res)
    #    id += 1
    print 'Done.'

if __name__ == "__main__":
    main()

# export CUDA_VISIBLE_DEVICES=0 python main.py /home/tianli/face_detection/results_one_person/original/google_beach_person /home/tianli/face_detection/results_one_person/original/forest_person


