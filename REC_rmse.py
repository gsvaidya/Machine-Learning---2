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
    plt.xlim(0,9)
    plt.plot(eps,accuracy,label = 'RMSE')
    plt.xlabel('epsilon(loss function)',fontsize=18)
    plt.ylabel('Accuracy',fontsize=18)
    plt.legend(loc=4)    
    plt.show()