
import numpy as np

X = lambda : np.random.uniform(0,1)
Y = lambda : np.random.uniform(0,2)

arr = np.array([ (X(),Y()) for _sample in range(100)])
print(arr)

var = np.var(arr,axis=0)
print(var)

