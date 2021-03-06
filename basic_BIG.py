# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 00:52:56 2016

@author: ARM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
import time
from keras.layers.advanced_activations import ELU
from sklearn.metrics import *
import matplotlib.pyplot as plt

lim=13000

name1='OFF_Back-tilt.txt'
cf1=pd.read_csv(name1,' ')
vals1=(cf1.iloc[1:lim,:])
vals1_n=(vals1-vals1.min(axis=0))/(vals1.max(axis=0)-vals1.min(axis=0))  #normalize

name2='OFF_Front-tilt.txt'
cf2=pd.read_csv(name2,' ')
vals2=(cf2.iloc[1:lim,:])
vals2_n=(vals2-vals2.min(axis=0))/(vals2.max(axis=0)-vals2.min(axis=0))  #normalize

name3='OFF_Left-tilt.txt'
cf3=pd.read_csv(name3,' ')
vals3=(cf3.iloc[1:lim,:])
vals3_n=(vals3-vals3.min(axis=0))/(vals3.max(axis=0)-vals3.min(axis=0))  #normalize

name4='OFF_Right-tilt.txt'
cf4=pd.read_csv(name4,' ')
vals4=(cf4.iloc[1:lim,:])
vals4_n=(vals4-vals4.min(axis=0))/(vals4.max(axis=0)-vals4.min(axis=0))  #normalize

name5='OFF_Upright.txt'
cf5=pd.read_csv(name5,' ')
vals5=(cf5.iloc[1:lim,:])
vals5_n=(vals5-vals5.min(axis=0))/(vals5.max(axis=0)-vals5.min(axis=0))  #normalize

name6='OFF_Upside-down.txt'
cf6=pd.read_csv(name6,' ')
vals6=(cf6.iloc[1:lim,:])
vals6_n=(vals6-vals6.min(axis=0))/(vals6.max(axis=0)-vals6.min(axis=0))  #normalize

name1a='ON_Back-tilt.txt'
af1=pd.read_csv(name1a,' ')
vals1a=(af1.iloc[1:lim,:])
vals1a_n=(vals1a-vals1a.min(axis=0))/(vals1a.max(axis=0)-vals1a.min(axis=0))  #normalize

name2a='ON_Front-tilt.txt'
af2=pd.read_csv(name2a,' ')
vals2a=(af2.iloc[1:lim,:])
vals2a_n=(vals2a-vals2a.min(axis=0))/(vals2a.max(axis=0)-vals2a.min(axis=0))  #normalize

name3a='ON_Left-tilt.txt'
af3=pd.read_csv(name3a,' ')
vals3a=(af3.iloc[1:lim,:])
vals3a_n=(vals3a-vals3a.min(axis=0))/(vals3a.max(axis=0)-vals3a.min(axis=0))  #normalize

name4a='ON_Right-tilt.txt'
af4=pd.read_csv(name4a,' ')
vals4a=(af4.iloc[1:lim,:])
vals4a_n=(vals4a-vals4a.min(axis=0))/(vals4a.max(axis=0)-vals4a.min(axis=0))  #normalize

name5a='ON_Upright.txt'
af5=pd.read_csv(name5a,' ')
vals5a=(af5.iloc[1:lim,:])
vals5a_n=(vals5a-vals5a.min(axis=0))/(vals5a.max(axis=0)-vals5a.min(axis=0))  #normalize

name6a='ON_Upside-down.txt'
af6=pd.read_csv(name6a,' ')
vals6a=(af6.iloc[1:lim,:])
vals6a_n=(vals6a-vals6a.min(axis=0))/(vals6a.max(axis=0)-vals6a.min(axis=0))  #normalize

clas_1=np.vstack([vals1_n])
clas_2=np.vstack([vals2_n])
clas_3=np.vstack([vals3_n])
clas_4=np.vstack([vals4_n])
clas_5=np.vstack([vals5_n])
clas_6=np.vstack([vals6_n])
clas_7=np.vstack([vals1a_n])
clas_8=np.vstack([vals2a_n])
clas_9=np.vstack([vals3a_n])
clas_10=np.vstack([vals4a_n])
clas_11=np.vstack([vals5a_n])
clas_12=np.vstack([vals6a_n])

a_lab=pd.DataFrame(np.zeros([len(clas_1)],dtype=int)) #indexing 0 to the class
b_lab=pd.DataFrame(np.zeros([len(clas_2)],dtype=int)+1) #indexing 1 to the class
c_lab=pd.DataFrame(np.zeros([len(clas_3)],dtype=int)+2)
d_lab=pd.DataFrame(np.zeros([len(clas_4)],dtype=int)+3)
e_lab=pd.DataFrame(np.zeros([len(clas_5)],dtype=int)+4)
f_lab=pd.DataFrame(np.zeros([len(clas_6)],dtype=int)+5)
g_lab=pd.DataFrame(np.zeros([len(clas_6)],dtype=int)+6)
h_lab=pd.DataFrame(np.zeros([len(clas_6)],dtype=int)+7)
i_lab=pd.DataFrame(np.zeros([len(clas_6)],dtype=int)+8)
j_lab=pd.DataFrame(np.zeros([len(clas_6)],dtype=int)+9)
k_lab=pd.DataFrame(np.zeros([len(clas_6)],dtype=int)+10)
l_lab=pd.DataFrame(np.zeros([len(clas_6)],dtype=int)+11)

a=clas_1
b=clas_2
c=clas_3
d=clas_4
e=clas_5
f=clas_6
g=clas_7
h=clas_8
i=clas_9
j=clas_10
k=clas_11
l=clas_12

#dat=pd.concat([a_dat,b_dat,c_dat,d_dat])
dat=np.vstack([a,b,c,d,e,f,g,h,i,j,k,l]) 
lab=pd.concat([a_lab,b_lab,c_lab,d_lab,e_lab,f_lab,g_lab,h_lab,i_lab,j_lab,k_lab,l_lab])
X=dat
y=(lab.values).ravel() #ravel used for formatting purpose


X_train, X_test, y_train, y_test = train_test_split(X, y)
y_test=np.eye(12)[y_test]
y_train=np.eye(12)[y_train]

import keras
model = Sequential()
act = keras.layers.advanced_activations.ELU(alpha=0.1)
model.add(Dense(2056, input_shape=(6,)))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation(act))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation(act))
model.add(Dropout(0.5))
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(12, activation='softmax'))
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
model.summary()
#adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
#model.compile(loss='mse', optimizer=adam)

model.fit(X_train, y_train, nb_epoch=1, batch_size=128, validation_split=0.3)
time.sleep(2)
y_pred=model.predict_classes(X_test)
y_pred=np.eye(12)[y_pred]

print(classification_report(y_test,y_pred))
print (accuracy_score(y_test,y_pred))

"""
conf_arr = confusion_matrix(y_test, y_pred)
print(conf_arr)

norm_conf = []
for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i,0)
        for j in i:
                tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

fig = plt.figure()
ax = fig.add_subplot(111)
res = ax.imshow(norm_conf, interpolation='nearest')
for i, cas in enumerate(conf_arr):
    for j, c in enumerate(cas):
        if c>0:
            plt.text(j-.2, i+.2, c, fontsize=14)
cb = fig.colorbar(res)
plt.title('Confusion matrix of the classifier')
ax.set_xticklabels([''] + labels,fontsize=13)
ax.set_yticklabels([''] + labels,fontsize=13)
plt.xlabel('Predicted',fontsize=16)
plt.ylabel('True',fontsize=16)
plt.savefig("confmat.png", format="png",dpi=400)
"""
