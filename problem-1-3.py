#Measuring deviation in beta vs sigma
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


#Measuring deviation in beta vs n
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


pl.plot([i/100 for i in range(1,100)],n_devi)
pl.xlabel("n")
pl.ylabel("Deviation in Beta")
pl.savefig("n vs deviation-4.png")
pl.show()


#Measuring cost function vs sigma
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



print(np.linspace(0,1,100).shape,n_devi.shape)
pl.plot(np.linspace(0,1,100),n_devi)
pl.xlabel("sigma")
pl.ylabel("Cost function")
pl.savefig("Cost function vs sigma-4.png")
pl.show()
print(n_devi.shape)
print((np.linspace(0,1,100).reshape(100,1)).shape)
print(stats.linregress(np.linspace(0,1,100),np.array(n_devi).reshape(100,)))


#Measuring cost function vs n
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


pl.plot([i for i in range(1,100)],n_devi)
pl.xlabel("n")
pl.ylabel("Cost function")
pl.savefig("Cost function vs n-4.png")
pl.show()
print(stats.linregress(np.linspace(1,100,99),n_devi.reshape(99,)))


