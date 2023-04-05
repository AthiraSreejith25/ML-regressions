import numpy as np
import pickle
import matplotlib.pyplot as pl
from scipy import stats



np.random.seed(3)

n = 50 #size of dataset
m = 2 #Dimension of dataset,(m+1) given in question


#Problem 1

def func(sigma,x_dim,beta_dim):
    beta = np.random.uniform(0,1,(beta_dim,1)) #Fixing beta values to random numbers between 0 and 1
    
    x = np.ones((x_dim,beta_dim)) #Initialising x matrix
    x[1:,:] = np.random.uniform(size = (x_dim-1,beta_dim)) #fixing all values in x, except the first column elements to random numbers between 0 and 1

    e = np.random.normal(0, sigma,size=(x_dim,1)) #normal random numbers with mean 0, and sigma standard deviation
    y = x.dot(beta) + e #the independent variable y

    return(beta,x,y)

print(func(0.2,n,m))
