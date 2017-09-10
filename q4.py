#!/usr/bin/env python

import sys
import os
import numpy as np

"""Feel free to add any extra classes/functions etc as and when needed.
This code is provided purely as a starting point to give you a fair idea
of how to go about implementing machine learning algorithms in general as
a part of the first assignment. Understand the code well"""

classes = ['galsworthy/','galsworthy_2/','mill/','shelley/','thackerey/','thackerey_2/','wordsmith_prose/','cia/','johnfranklinjameson/','diplomaticcorr/']

classes_output = ['galsworthy','galsworthy_2','mill','shelley','thackerey','thackerey_2','wordsmith_prose','cia','johnfranklinjameson','diplomaticcorr']

class FeatureVector(object):
    def __init__(self,vocabsize,numdata):
        self.vocabsize = vocabsize
        self.X =  np.zeros((numdata,self.vocabsize), dtype=np.int)
        self.Y =  np.zeros((numdata,), dtype=np.int)

    def make_featurevector(self, input, classid):
        pass

class KNN(object):
    def __init__(self,trainVec,testVec):
        self.X_train = trainVec.X
        self.Y_train = trainVec.Y
        self.X_test = testVec.X
        self.Y_test = testVec.Y
        self.metric = Metrics('accuracy')

    def classify(self, nn=1):
        y_pred = [0]*self.Y_test.shape[0]
        for i in range(self.Y_test.shape[0]):
            Y_pred = classes[np.random.randint(0,10)]
            vec = self.X_test[i]
            dist = (self.X_train-vec)**2
            dist2 = dist.sum(axis=1)
            vals = dist2.argsort()[:nn]
            f = [0]*10
            for  j in range(0,nn):
                val = vals[j]
                val = self.Y_train[val]
                f[val]=f[val]+1
            indx = f.index(max(f))
            y_pred[j]=indx
            print classes_output[y_pred[j]]
        #for i in range(self.Y_test.shape[0]):
        #    Y_pred[i] = classes[y_pred[i]]
        #print Y_pred
        #print(Y_pred.strip('/'))

class Metrics(object):
    def __init__(self,metric):
        self.metric = metric

    def score(self):
        if self.metric == 'accuracy':
            return self.accuracy()
        elif self.metric == 'f1':
            return self.f1_score()

    def get_confmatrix(self,y_pred,y_test):
            """
                Implements a confusion matrix
                """

    def accuracy(self):
            """
                Implements the accuracy function
                """

    def f1_score(self):
            """
                Implements the f1-score function
                """

if __name__ == '__main__':
    traindir = sys.argv[1]
    testdir = sys.argv[2]
    inputdir = [traindir,testdir]

    vocab = 0 #Random Value
    trainsz = 0 #Random Value
    testsz = 0 #Random Value
    dictionary = {}
    for idir in inputdir:
        for c in classes:
            listing = os.listdir(idir+c)
            for filename in listing:
                if idir == testdir:
                    testsz = testsz+1
                else:
                    trainsz = trainsz+1
                f = open(idir+c+filename,'r')
                for line in f:
                    words = line.split()
                    words = words[1:-1]
                    for word in words:
                        if word not in dictionary:
                            vocab= vocab+1
                            dictionary[word]=vocab-1
                

        # print('Making the feature vectors.')
    trainVec = FeatureVector(vocab,trainsz)
    testVec = FeatureVector(vocab,testsz)

    for idir in inputdir:
        classid = 0
        cnt=-1
        for c in classes:
            listing = os.listdir(idir+c)
            for filename in listing:
                cnt=cnt+1
                f = open(idir+c+filename,'r')
                for line in f:
                    words = line.split()
                    words = words[1:-1];
                    for word in words:
                        if idir == traindir:
                            trainVec.X[cnt][dictionary[word]]=trainVec.X[cnt][dictionary[word]]+1
                        else:
                            testVec.X[cnt][dictionary[word]]=testVec.X[cnt][dictionary[word]]+1
                if idir == traindir:
                    trainVec.Y[cnt]=classid
                else:
                    testVec.Y[cnt]=classid
                
            #if idir == traindir:
            #    trainVec.make_featurevector(inputs,classid)
            #else:
            #    testVec.make_featurevector(inputs,classid)
            classid += 1

        # print('Finished making features.')
        # print('Statistics ->')

    knn = KNN(trainVec,testVec)
    knn.classify()
