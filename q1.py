import numpy as np
import sys
import csv
from io import BytesIO

train_file = sys.argv[1]
test_file = sys.argv[2]

def checking(y_test,yn,cnt_test):
    temp2 = 0
    for i in range(0,cnt_test):
        if(yn[i]==y_test[i]):
            temp2 = temp2+1
    disp = [temp2,cnt_test]
    #print(disp)
    return


def prediction(x_test,y_test,w,cnt_test):
    yn = [0]*cnt_test
    
    for i in range(0,cnt_test):
        w2 = w
        np.transpose(w2)
        b = x_test[i]
        c = np.dot(w2,b)
        if(c<0):
            yn[i]=0
        else:
            yn[i]=1
        print(yn[i])
    return [yn]


def algo1(x_train,y_train,w,cnt_train):
    flag=1
    it=0
    while(flag):
        it=it+1
        #print(it)
        flag=0
        for i in range(0,cnt_train):
            w2 = w
            np.transpose(w2)
            b = x_train[i]
            if(y_train[i]==0):
                b=b*-1
            c = np.dot(w2,b)
            if(c<=0):
                w = np.add(w,b)
                flag=1
    return [w]

def algo2(x_train,y_train,w,cnt_train):
    flag= 1
    b = 5
    it = 0
    while(flag):
        it = it+1
        #print(it)
        flag = 0
        for i in range(0,cnt_train):
            w2 = w
            np.transpose(w2)
            temp = x_train[i]
            if(y_train[i]==0):
                temp = temp*-1
            c = np.dot(w2,temp)
            if(c-b<=0):
                w = np.add(w,temp)
                flag=1

    return [w]

def algo3(x_train,y_train,w,cnt_train):
    flag = 1
    it = 0
    while(flag):
        change = np.zeros(28*28+1)
        it = it+1
        flag = 0
        #print(it)
        for i in range(0,cnt_train):
            w2 = w
            np.transpose(w2)
            temp = x_train[i]
            if(y_train[i]==0):
                temp = temp*-1
            c = np.dot(w2,temp)
            if(c<=0):
                change = np.add(change,temp)
                flag=1

        w = np.add(w,change)

    return [w]

def algo4(x_train,y_train,w,cnt_train):
    flag = 1
    b = 5
    it = 0
    while(flag):
        change = np.zeros(28*28+1)
        it = it+1
        flag = 0
        #print(it)
        for i in range(0,cnt_train):
            w2 = w
            np.transpose(w2)
            temp = x_train[i]
            if(y_train[i]==0):
                temp = temp*-1
            c = np.dot(w2,temp)
            if(c<=b):
                change = np.add(change,temp)
                flag=1

        w = np.add(w,change)

    return [w]



x_train = []
y_train = []
x_test = []
y_test = []
cnt_test = 0
cnt_train = 0
with open(train_file,'r') as f:
    reader = csv.reader(f)
    for row in reader:
        cnt_train = cnt_train+1
        y_train = y_train + [int(row[0])]
        b =  [1]
        b = b + (row[1:28*28+1])*1
        x_train = x_train + [b]

with open(test_file,'r') as f:
    reader = csv.reader(f)
    for row in reader:
        cnt_test = cnt_test+1
        #y_test = y_test+[int(row[0])]
        b = [1]
        b = b + (row)*1
        x_test = x_test + [b]

x_train = np.array(x_train,dtype = float)
x_test = np.array(x_test,dtype = float)

w = np.zeros(28*28+1)
[w] = algo1(x_train,y_train,w,cnt_train)
[yn] = prediction(x_test,y_test,w,cnt_test)
#checking(y_test,yn,cnt_test)

w = np.zeros(28*28+1)
[w] = algo2(x_train,y_train,w,cnt_train)
[yn] = prediction(x_test,y_test,w,cnt_test)
#checking(y_test,yn,cnt_test)

w = np.zeros(28*28+1)
[w] = algo3(x_train,y_train,w,cnt_train)
[yn] = prediction(x_test,y_test,w,cnt_test)
#checking(y_test,yn,cnt_test)

w = np.zeros(28*28+1)
[w] = algo4(x_train,y_train,w,cnt_train)
[yn] = prediction(x_test,y_test,w,cnt_test)
#checking(y_test,yn,cnt_test)




