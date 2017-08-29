import numpy as np
import csv
import sys
from math import *

train_file = sys.argv[1]
test_file = sys.argv[2]

x_train = []
y_train = []
sales = ['sales','accounting','technical','management','support','IT','product_mng','hr','marketing','RandD']
salary = ['low','medium','high']
cnt_train = 0
flag = 0
x_test = []
y_test = []
cnt_test  = 0

class Node(object):
    def __init__(self, data1,data2,cnt):
        self.data1 = data1
        self.data2 = data2
        self.cntdata = cnt
        self.children = []
        self.val=-1
        self.leaf = 0
        self.index = -1
        self.type = -1
        self.f_check = [0]*9
        self.vals_f = [0]*9

    def add_child(self, obj):
        self.children.append(obj)

with open(train_file,'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if flag==0:
            flag = 1
            continue
        else:
            cnt_train = cnt_train+1
            y_train = y_train+[int(row[6])]
            row[8]=sales.index(row[8])
            row[9]=salary.index(row[9])
            row = row[0:6]+row[7:]
            x_train = x_train + [row]

flag=0
with open(test_file,'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if flag==0:
            flag = 1
            continue
        else:
            cnt_test = cnt_test + 1
            
            row[7] = sales.index(row[7])
            row[8] = salary.index(row[8])
            x_test = x_test + [row]

x2_train = []
x2_test = []
for i in range(0,cnt_train):
    x2_train = x2_train + [map(float,x_train[i])]
for i in range(0,cnt_test):
    x2_test = x2_test + [map(float,x_test[i])]
x_train = x2_train
x_test = x2_test

def entropy(ar1,ar2,flag,cnt_train):
    fp=0.0
    fn=0.0
    for j in range(0,cnt_train):
        if ar1[j]==flag and ar2[j]==1:
            fp = fp+1
        elif ar1[j]==flag and ar2[j]==0:
            fn = fn+1
    if fp+fn==0:
        return 1000;
    pp = fp*1.0/(fp+fn)
    pn = fn*1.0/(fp+fn)
    if pp==0:
        e = pn*log(pn,2)
    elif pn==0:
        e = pp*log(pp,2)
    else:
        e = pp*log(pp,2) + pn*log(pn,2)
    e = -1.0*e
    return e

def qvaltrain(flag,index,x_train,y_train,cnt_train,itcnt):
    g_val=0;
    if flag==1:
        val=1
        l = [row[index] for row in x_train]
        minval = min(l)
        maxval = max(l)
        q=['Inf']*9
        for i in range(1,10):
            ans=0.0
            val = minval+(i*1.0/10)*(maxval-minval)
            f2 = [0]*cnt_train
            c2= [0]*itcnt
            for j in range(0,cnt_train):
                if(x_train[j][index]>val):
                    f2[j]=1
                    c2[1]=c2[1]+1
                else:
                    c2[0]=c2[0]+1
            for j in range(0,itcnt):
                ans = ans+entropy(f2,y_train,j,cnt_train)*1.0*(c2[j]*1.0/cnt_train)
            q[i-1]=ans
        indxs = q.index(min(q))
        val = minval + ((indxs+1)*1.0/10)*(maxval-minval)
        return [min(q),val]
        
    else:
        ans = 0.0
        l = [row[index] for row in x_train]
        #f2 = [0]*cnt_train
        c2 = [0]*itcnt
        for j in range(0,cnt_train):
            c2[int(x_train[j][index])]=c2[int(x_train[j][index])]+1

        for j in range(0,itcnt):
            ans = ans+ entropy(l,y_train,j,cnt_train)*1.0*(c2[j]*1.0/cnt_train)
        return [ans,0]


f = [1,1,0,1,0,0,0,0,0]
f_check = [0,0,0,0,0,0,0,0,0]
vals_f =  [0,0,0,0,0,0,0,0,0]
itc = [2,2,20,2,15,2,2,10,3]
flag_stop=1
cnt_comp=0
root = Node(x_train,y_train,cnt_train)
for j in range(0,9):
    root.f_check[j]=f_check[j]
    root.vals_f[j]= vals_f[j]
yn = [0]*cnt_test

def check(node,row):
    global f
    global flag_stop
    global cnt_comp
    global itc

    if node.leaf==1:
        #print [node.index,node.type]
        return node.type
    else:
        if f[node.index]==1:
            temp = row[node.index]
            if temp>node.vals_f[node.index]:
                t = 1
            else:
                t = 0
        else:
            t = row[node.index]
        for i in range(0,itc[node.index]):
            k = node.children[i]
            if k.val==t:
                #print node.index,
                return check(k,row)
    

def algo(node):
    global f
    global flag_stop
    global cnt_comp
    global itc

    flag=1
    for i in range(0,9):
        if node.f_check[i]==0:
            flag=0
            break
    if flag==1:
        flag_stop=0
        fp = sum(i==1 for i in node.data2)
        fn = sum(i==0 for i in node.data2)
        if fp>=fn:
            node.type=1
            node.leaf =1
            node.index = -4
        else:
            node.type=0
            node.leaf =1
            node.index = -4
        return
    
    q2 = [0]*9
    vals = [0]*9
    for i in range(0,9):
        if node.f_check[i]==1:
            q2[i] = float('Inf')
            continue
        else:
            [q2[i],vals[i]]=qvaltrain(f[i],i,node.data1,node.data2,node.cntdata,itc[i]);
    ind = q2.index(min(q2))
    if f[ind]==1:
        node.vals_f[ind] = vals[ind]
    node.index = ind;
    node.f_check[ind]=1
    for i in range(0,itc[ind]):
        data = []
        data2 = []
        cnt=0
        fp=0
        fn=0
        for j in range(0,node.cntdata):
            if f[ind]==1:
                if(i==0 and node.data1[j][ind]<=node.vals_f[ind]):
                    data = data + [node.data1[j]]
                    data2 = data2 + [node.data2[j]]
                    if node.data2[j]==1:
                        fp=fp+1
                    else:
                        fn = fn+1
                    cnt = cnt+1
                elif(i==1 and node.data1[j][ind]>node.vals_f[ind]):
                    data = data + [node.data1[j]]
                    data2 = data2 + [node.data2[j]]
                    cnt = cnt+1
                    if node.data2[j]==1:
                        fp = fp+1
                    else:
                        fn = fn+1
            else:
                if(node.data1[j][ind]==i):
                    data = data + [node.data1[j]]
                    data2 = data2 + [node.data2[j]]
                    cnt = cnt +1
                    if node.data2[j]==1:
                        fp =fp+1
                    else:
                        fn = fn+1

        m = Node(data,data2,cnt)
        
        for j in range(0,9):
            m.f_check[j]=node.f_check[j]
            m.vals_f[j]=node.vals_f[j]

        m.val=i
        if fn==0 or fp==0:
            m.leaf = 1
            if fp==0:
                m.type=0
            else:
                m.type=1
        node.add_child(m)

    for i in range(0,itc[ind]):
        k = node.children[i]
        if k.leaf==1:
            continue
        else:
            algo(k)
    return
        

algo(root)


for i in range(0,cnt_test):
    row = x_test[i]
    yn[i]=check(root,row)
    print yn[i]
#print yn

    








            
                

                





