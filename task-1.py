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



#Problem 2

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


def sig_dev(sig, iterations):
    deviations = []
    dev = 0
    for j in range(sig):
       
        sigma = j/sig
        b, axe, aye = func(sigma,n,m)
        print(j)
        for i in range(iterations):
            dev += (1/n)*np.linalg.norm(b - func2(axe,aye,10000,10**(-5))[0])
        deviations.append(dev/iterations)
    return(np.array(deviations))


sigma_dev = sig_dev(100,50)

with open('sigma_deviations-1.txt', 'wb') as fh:
   pickle.dump(sigma_dev, fh)

 
pl.plot([i/100 for i in range(100)],sigma_dev)
pl.xlabel("Sigma")
pl.ylabel("Deviation in Beta")
pl.savefig("Sigma vs deviation-4.png")
pl.show()

def n_dev(n_values, iterations):
    deviations = []
    dev = 0
    for j in range(1,n_values):
        
        n = j
        b, axe, aye = func(0.5,n,m)
        print(j)
        
        for i in range(iterations):
            dev += (1/n)*np.linalg.norm(b - func2(axe,aye,10000,10**(-5))[0])
        deviations.append(dev/iterations)
    return(np.array(deviations))

n_devi = n_dev(100,50)



with open('n_deviations-1.txt', 'wb') as fh:
   pickle.dump(n_devi, fh)

file = open('n_deviations-1.txt', 'rb')
n_devi = pickle.load(file)


pl.plot([i/100 for i in range(1,100)],n_devi)
pl.xlabel("n")
pl.ylabel("Deviation in Beta")
pl.savefig("n vs deviation-4.png")
pl.show()

def sigma_dev_costf(sigma, iterations):
    deviations = []
    dev = 0
    for j in range(sigma):
        
        b, axe, aye = func(sigma,n,m)
        print(j)
        
        for i in range(iterations):
            dev += cost_f(axe,b,aye)
        deviations.append(dev/iterations)
    return(np.array(deviations))

sigma_dev_cost = sigma_dev_costf(100,50)

with open('n_deviations.txt', 'wb') as fh:
   pickle.dump(sigma_dev_cost, fh)

file = open('n_deviations.txt', 'rb')
n_devi = pickle.load(file)

print(np.linspace(0,1,100).shape,n_devi.shape)
pl.plot(np.linspace(0,1,100),n_devi)
pl.xlabel("sigma")
pl.ylabel("Cost function")
pl.savefig("Cost function vs sigma-4.png")
pl.show()
print(n_devi.shape)
print((np.linspace(0,1,100).reshape(100,1)).shape)
print(stats.linregress(np.linspace(0,1,100),np.array(n_devi).reshape(100,)))


def n_dev_costf(n_values, iterations):
    deviations = []
    dev = 0
    for j in range(1,n_values):
        
        n = j
        b, axe, aye = func(0.5,n,m)
        print(j)
        
        for i in range(1,iterations):
            dev += cost_f(axe,b,aye)
        deviations.append(dev/iterations)
    return(np.array(deviations))

n_dev_cost = n_dev_costf(100,50)

with open('n_deviations.txt', 'wb') as fh:
   pickle.dump(n_dev_cost, fh)

file = open('n_deviations.txt', 'rb')
n_devi = pickle.load(file)


pl.plot([i for i in range(1,100)],n_devi)
pl.xlabel("n")
pl.ylabel("Cost function")
pl.savefig("Cost function vs n-4.png")
pl.show()
print(stats.linregress(np.linspace(1,100,99),n_devi.reshape(99,)))

    

    

