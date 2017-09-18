import numpy as np
import sys
import csv

train_file = sys.argv[1]
test_file = sys.argv[2]


def prediction(x_test,y_test,w,cnt_test):
    yn = np.zeros(cnt_test)
    for i in range(0,cnt_test):
        c = np.dot(w,x_test[i])
        if(c<0):
            yn[i]=2
        else:
            yn[i]=4
        print int(yn[i])
    return [yn]

def print_val(yn,y_test,cnt_test):
    temp = 0
    for i in range(0,cnt_test):
        if(yn[i]==y_test[i]):
            temp=temp+1
        
    return ((temp*(1.0))/cnt_test)



def algo1(x_train,y_train,cnt_train,w):
    it = 0
    b = 0.5
    flag = 1
    while(flag==1):
        it=it+1
        #print(it)
        if(it==100):
            break
        flag=0
        #temp_val = int(round(0.8*cnt_train))
        temp_val = cnt_train
        for i in range(0,temp_val):
            temp = np.zeros(10)
            w2 = w
            temp2 = x_train[i]
            if(y_train[i]==2):
                temp2 = -1*temp2
            np.transpose(w2)
            c = np.dot(w2,temp2)
            if(c<=b):
                d = np.dot(temp2,temp2)
                temp2 = (c-b)*temp2;
                temp2 = (1.0/d)*temp2;
                temp = np.add(temp,temp2)
                flag = 1
            w = np.subtract(w,temp)
    return [w]
            
def algo2(x_train,y_train,cnt_train,w):
    it = 0
    eta = 1
    flag = 1
    while(flag==1):
        it = it+1
        eta = 1/(1.0*it)
        #print(it)
        if(it==100):
            break
        flag = 0
        #temp_val = int(round(0.8*cnt_train))
        temp_val = cnt_train
        for i in range(0,temp_val):
            w2 = w
            temp2 = x_train[i]
            if(y_train[i]==2):
                temp2 = -1*temp2
            np.transpose(w2)
            c = np.dot(w2,temp2)
            if(c<=0):
                temp2 = eta *temp2
                flag = 1
                w = np.add(w,temp2)
    return [w]


x_train = []
y_train = []
id_train = []
x_test = []
y_test = []
id_test = []
cnt_test = 0
cnt_train = 0

with open(train_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        flag = 0
        for i in range(0,len(row)):
            if(row[i]=='?'):
                flag = 1
                break
        if flag==1:
            continue
        cnt_train = cnt_train + 1
        y_train = y_train + [int(row[10])]
        id_train = id_train + [int(row[0])]
        b = [1]
        b = b+ (row[1:10])*1
        x_train = x_train + [b]

with open(test_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        flag = 0
        for i in range(0,len(row)):
            if(row[i]=='?'):
                flag = 1
                break
        if flag == 1:
            continue
        cnt_test = cnt_test + 1
        #y_test = y_test + [int(row[10])]
        id_test = id_test + [int(row[0])]
        b = [1]
        b = b + (row[1:])*1
        x_test = x_test + [b]

x_train = np.array(x_train,dtype = float)
x_test = np.array(x_test,dtype = float)

w = np.zeros(10)
[w] = algo1(x_train,y_train,cnt_train,w)
[yn] = prediction(x_test,y_test,w,cnt_test)
#accuracy = print_val(yn,y_test,cnt_test)
#print(accuracy)

w = np.zeros(10)
[w] = algo2(x_train,y_train,cnt_train,w)
[yn] = prediction(x_test,y_test,w,cnt_test)
#accuracy = print_val(yn,y_test,cnt_test)
#print(accuracy)

