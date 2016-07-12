import numpy as np
import math as ma
import matplotlib.pyplot as plt

    
def rr(X_red,lamb,y):
    temp = 0
    X_t=X_red.transpose()
    p1 = np.dot(X_t,X_red)
    print np.shape(p1)
    p2 = np.identity(len(p1))
    
    p3 = np.dot(lamb,p2)
    temp = np.linalg.inv(p1-p3)
    w = np.dot(np.dot(temp,X_t),y)
    return w

def standardize(origX):
    means = origX.mean(axis=0)
    stds = origX.std(axis=0)
    return (origX - means) / stds

def rmse(X,y,w):
    temp = 0
    h = np.zeros(len(X))
    for i in range(len(X)):
        h[i] = np.dot(w.transpose(),X[i]) 
        temp +=((y[i] - h[i])**2)
    return ma.sqrt(temp/len(X))
            
def mad(X,y,w):
    temp=0
    h = np.zeros(len(X))
    for i in range(len(X)):
        h[i] = np.dot(w.transpose(),X[i])
        temp +=abs(y[i] - h[i])
    return temp/len(X)        

#def plot(X,y):
#    lamb = []
#    for i in range(-10,10,2):
#        lamb.append(lamb[i])
#        j = rr(X_red,lamb[i],y_red)
#        k = rmse(X_red,y_red,j)
#        l = mad(X_red,y_red,j)
#        plt.plot(lamb[i],k)
#        plt.xscale('log')
#        plt.show()
#        plt.plot(lamb[i],l)
#        plt.xscale('log')
#        plt.show()
        

def plot(X_train,y_train,X_test,y_test):
    lamb = np.zeros(20)
    r = np.zeros(20)
    m = np.zeros(20)
    j = 0
    for i in range(-10,10):
        lamb[j]=10**i
        w = rr(X_train,lamb[j],y_train)
        r[j] = rmse(X_test,y_test,w)
        m[j] = mad(X_test,y_test,w)
        j = j + 1
        
    print lamb
    plt.plot(lamb,r)
    plt.xscale('log',basex=10)
    plt.xlabel('lamda',fontsize=18)
    plt.ylabel('',fontsize=18)
    plt.show()
    plt.plot(lamb,m)
    plt.xscale('log',basex=10)
    plt.show()    

    
def REC(X,y,w):
    j = 0
    k = 0
    eps = np.zeros(20)
    accuracy = np.zeros(20)
    h = np.zeros(len(X))
    for j in np.arange(0,4,0.2):
        eps[k] = j
        hit = 0.0
        for i in range(len(X)):
                          
            h[i] = np.dot(w.transpose(),X[i])
            
        
            if abs((h[i]-y[i])<=eps[k]):
                hit = hit + 1
        
        accuracy[k] = hit/len(X)
        k=k+1
    print accuracy 
    plt.plot(eps,accuracy)
    plt.show()

def pearson(X,y):
    cov = np.cov(X,y)
    s1 = standardize(X)
    s2 = standardize(y)
    pm = cov/(s1*s2)
    return pm
    
if __name__=='__main__' :
    X=np.genfromtxt("red.txt", delimiter=";")
    X_red = X[:,0:11]
    y_red = X[:,11]
    a = int(0.8*len(X))
    
    X_white=np.genfromtxt("white.txt", delimiter=";")
    #print 'X_red is', X_red
    #print 'X_white is',X_white
    #print 'y_red is',y_red
    X_red = standardize(X_red)
    y_red = standardize(y_red)
    w = rr(X_red,1,y_red)
    print w
    X_train = X_red[0:a,:]
    y_train = y_red[0:a]
    X_test = X_red[a:,:]
    y_test = y_red[a:]
    rm = rmse(X_train,y_train,w)
    print 'rmse',rm
    m = mad(X_train,y_train,w)
    print 'mad',m   
    #p = plot(X_train,y_train,X_test,y_test)
    r2 = REC(X_test,y_test,w)
    pcc = pearson(X_test,y_test)