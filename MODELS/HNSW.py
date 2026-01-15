import numpy as np
import pandas as pd 
import random
print("Import successful")

#INITIALIZE RANDOM 2-d array with size (100*2)
arr=[]
for i in range(100):
    temp=[]
    for j in range(2):
        temp.append(random.randint(-1000,1000))
    arr.append(temp)

def fun(arr,)