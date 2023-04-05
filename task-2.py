import numpy as np
import pickle
import matplotlib.pyplot as pl
from scipy import stats



np.random.seed(10)

n = 50 #size of dataset
m = 2 #Dimension of dataset,(m+1) given in question


#Problem 1

def func(theta,x_dim,beta_dim):
    beta = np.random.normal(0,2,size = (beta_dim,1)) #Fixing beta values to random numbers between 0 and 1
    
    x = np.ones((x_dim,beta_dim)) #Initialising x matrix
    x[:,1:] = np.random.uniform(size = (x_dim,beta_dim-1)) #fixing all values in x, except the first column elements to random numbers between 0 and 1

    #e = np.random.normal(0, sigma,size=(x_dim,1)) #normal random numbers with mean 0, and sigma standard deviation
    #the independent variable y
    #y = [f(x) if condition else g(x) for x in sequence]
    #y = [1 if (1/(1 + np.exp(-np.dot(x,beta))))]
    #print(x.shape,len(beta[0]))
    #y = [1 if (1/(1 + np.exp(-np.dot(x[i],beta)))) > 0.5 else 0 for i in range(len(x))]
    y = np.array([1 if (1/(1 + np.exp(-np.dot(i,beta)))) > 0.5 else 0 for i in x])
    print(y)
    z = np.array([0 if np.random.uniform(0,1) > theta else 1 for i in range(len(y))])
    print(z)
    
    
    return(beta,x,(y+z)%2)



b,X,Y = func(0.4,2,2)
"""
#Cost function
def cost_f(x,beta,y): 
    l = np.dot(x,beta) - y
    return(sum(l**2)/len(x))


#function that learns the parameters of a linear regression line given the inputs as shown
def func2(x,y,k,tau,learning = 0.001):
    beta = np.array(np.random.uniform(size = (x.shape[1],1)))
    cost = cost_f(x,beta,y)
    
    condition = True
    for ep in range(k):
        while condition:
            beta -= learning*np.dot(x.T,(np.dot(x,beta) - y))*(2/n)
            
            cost_new = cost_f(x,beta,y)
            
            condition = abs(cost_new - cost) > tau
            cost = cost_new

            if ep == k-1:
                print("Exceeded maximum number of iterations")
            
            
    return(beta,cost)
"""
#function that learns the parameters of a linear regression line given the inputs as shown
def func2(x,y,k,tau,learning = 0.001):
    beta = np.array(np.random.uniform(size = (x.shape[1],1)))
    cost = cost_f(x,beta,y)
    
    condition = True
    for ep in range(k):
        while condition:
            beta -= learning*np.dot(x.T,(np.dot(x,beta) - y))*(2/n)
            
            cost_new = cost_f(x,beta,y)
            
            condition = abs(cost_new - cost) > tau
            cost = cost_new

            if ep == k-1:
                print("Exceeded maximum number of iterations")
            
    return(beta,cost)

b_est, c = func2(X,Y,1000,10**(-4))
print("Estimated beta = ",b_est,"Cost function = ",c)



def cost_f(x,beta,y): 
    l = np.array([y[i]*np.log(1/(1 + np.exp(-np.dot(x[i],beta)))) + (1 -y[i])*np.log(np.exp(-np.dot(x[i],beta))/(1 + np.exp(-np.dot(x[i],beta)))) for i in range(len(y))])
    #print(l)
    #print(np.log(1/(1 + np.exp(-np.dot(x,beta)))))
    #print(y*np.log(1/(1 + np.exp(-np.dot(x,beta)))))
    return(-sum(l)/len(y))

print(cost_f(X,b,Y))



