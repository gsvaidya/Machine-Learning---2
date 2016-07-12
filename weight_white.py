#RidgeRegression_white
import numpy as np
import math as ma
import matplotlib.pyplot as plt
    
def rr(X_red,lamb,y):
    temp = 0
    X_t=X_red.transpose()
    p1 = np.dot(X_t,X_red)
    p2 = np.identity(len(p1))
    
    p3 = np.dot(lamb,p2)
    temp = np.linalg.inv(p1+p3)
    w = np.dot(np.dot(temp,X_t),y)
    return w
    

def standardize(origX):
    means = origX.mean(axis=0)
    stds = origX.std(axis=0)
    return (origX - means) / stds

def rmse(X,y,w):
    temp = 0
    print 'x',X
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

def plot(X_train,y_train,X_test,y_test):
    lamb = np.zeros(20)
    r = np.zeros(20)
    m = np.zeros(20)
    j = 0
    for i in range(-6,14):
        lamb[j]=10**i
        w = rr(X_train,lamb[j],y_train)
        r[j] = rmse(X_test,y_test,w)
        m[j] = mad(X_test,y_test,w)
        j = j + 1
        
    plt.plot(lamb,r,label='RMSE')
    plt.xscale('log',basex=10)
    plt.xlabel('lamda',fontsize=18)
    plt.ylabel('RMSE and MAD',fontsize=18)
    plt.show()
    plt.plot(lamb,m, label='MAD')
    plt.xscale('log',basex=10)
    plt.legend(loc=2)
    plt.show()    
    
def weightremoval(w,lamb,y_train,X_train,X_test,y_test):
    rm = np.zeros(11)
    ma = np.zeros(11)
    ind = []
    values = np.arange(11,0,-1)
    for i in range(0,11):
        w = rr(X_train,lamb,y_train)
        
        if (w!=[]):
            j=(np.argmin(abs(w)))
            print j
        ind.append(j)
        rm[i] = rmse(X_test,y_test,w)
        ma[i] = mad(X_test,y_test,w)
        X_train = np.delete(X_train,j,1)
        X_test = np.delete(X_test,j,1)
        print 'X',np.shape(X_train)
        
    plt.plot (values,rm, label='RMSE')
    plt.xlabel('no.of features',fontsize='18')
    plt.ylabel('RMSE',fontsize='18')
    plt.show()
    plt.plot (values,ma, label='MAD')
    plt.xlabel('no.of features',fontsize='18')
    plt.ylabel('MAD',fontsize='18')
    plt.legend(loc=1)
    plt.show()
            
    return ind
#def REC(X,y,w):
#    j = 0
#    k = 0
#    eps = np.zeros(20)
#    accuracy = np.zeros(20)
#    h = np.zeros(len(X))
#    for j in np.arange(0,4,0.2):
#        eps[k] = j
#        hit = 0.0
#        for i in range(len(X)):
#                          
#            h[i] = np.dot(w.transpose(),X[i])
#            
#            loss = abs(h[i]-y[i])
#            if (loss<=eps[k]):
#                hit = hit + 1
#        
#        accuracy[k] = hit/len(X)
#        k=k+1 
#    plt.plot(eps,accuracy)
#    plt.show()
def REC_mad(X,y,w):
    h = np.zeros(len(X))
    loss = np.zeros(len(X))
    for i in range(len(X)):
        h[i] = np.dot(w.transpose(),X[i])
        loss[i] = np.abs(h[i]-y[i])
    loss2 = np.sort(loss)
    eps=[]
    eps2 = 0.0
    hit=0.0
    accuracy =[]
    for i in range(len(X)):
        if (loss2[i]>eps2):
            accuracy.append(hit/len(X))
            eps2 = loss2[i]
            eps.append(eps2)
        hit = hit + 1
    accuracy.append(hit/len(X))
    eps.append(eps2)
    print 'eps',eps
    #plt.xlim(0,15)
    plt.plot(eps,accuracy,label='MAD')
    plt.xlabel('epsilon(loss function)',fontsize=18)
    plt.ylabel('Accuracy',fontsize=18)
    plt.legend(loc=4)
    plt.show()

def REC_rmse(X,y,w):
    h = np.zeros(len(X))
    loss = np.zeros(len(X))
    for i in range(len(X)):
        h[i] = np.dot(w.transpose(),X[i])
        loss[i] = (h[i]-y[i])**2
    loss2 = np.sort(loss)
    eps=[]
    eps2 = 0.0
    hit=0.0
    accuracy =[]
    for i in range(len(X)):
        if (loss2[i]>eps2):
            accuracy.append(hit/len(X))
            eps2 = loss2[i]
            eps.append(eps2)
        hit = hit + 1
    accuracy.append(hit/len(X))
    eps.append(eps2)
    print 'eps',eps
    #plt.xlim(0,20)
    plt.plot(eps,accuracy,label='RMSE')
    plt.xlabel('epsilon(loss function)',fontsize=18)
    plt.ylabel('Accuracy',fontsize=18)
    plt.show()
          
def pearson(X,y,w):
    pm = np.zeros(len(X[0]))
    s1 = np.std(X,0)    
    s2 = np.std(y)
    for i in range(len(X[0])):
        cv = np.cov(X[:,i],y)
        
        pm[i] = (cv[0][1])/s1[i]*s2
    plt.xlabel('weight vector',fontsize=18)
    plt.ylabel('Pearson correlation co-efficient',fontsize=18)
    plt.scatter(pm,w)
    plt.show()
    print 'pm',pm,'w',w
    print np.amax(pm),np.amax(w)
if __name__=='__main__' :
    X=np.genfromtxt("red.txt", delimiter=";")
    X_red = X[:,0:11]
    y_red = X[:,11]
    X2=np.genfromtxt("white.txt", delimiter=";")
    X_white = X2[:,0:11]
    y_white = X2[:,11]
    a = int(0.8*len(X2))
    X_white = standardize(X_white)
    y_white = standardize(y_white)
    X_train = X_white[0:a,:]
    y_train = y_white[0:a]
    w = rr(X_train,100000,y_train)    
    X_test = X_white[a:,:]
    y_test = y_white[a:]
    #z = weightremoval(w,100,y_train,X_train,X_test,y_test)
    #print z
    #rm = rmse(X_test,y_test,w)
    #print 'rmse',rm
    #m = mad(X_test,y_test,w)
    #print 'mad',m   
    #p = plot(X_train,y_train,X_test,y_test)
    #r2 = REC_rmse(X_test,y_test,w)
    #r3 = REC_mad(X_test,y_test,w)
    pmmc = pearson(X_train,y_train,w)
    #print 'pmmc',pmmc