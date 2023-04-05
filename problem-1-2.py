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
