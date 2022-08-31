# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 09:14:13 2022

@author: Redwoods
"""
a=1
b=2
c=a+b
print(c)

def add(a, b):
    return a+b

c=add(a,b)
print(c)

x=[1,2,3]
y=[4,5,6]

z=add(x,y)
print(z)  

import matplotlib.pyplot as plt
plt.plot(a,b,'ro', ms=16)
# plt.show()
plt.plot(x,y, 'bo', ms=16)
# plt.show()

##########################
def c2f(c):
    f = 5*(c-32)/9
    return f

c2f(-40)

# c2f([-20,0,20,32])  => Error!

def c2fn(c):
    fn=list()
    for tc in c:
        f = 5*(tc-32)/9
        fn.append(f)
    return fn

c2fn([-20,0,20, 32])

#
# Py modules
#
import tensorflow
tensorflow.__version__

import keras
keras.__version__

import scikeras
scikeras.__version__

import sklearn
sklearn.__version__

