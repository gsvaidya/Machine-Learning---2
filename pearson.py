def pearson(X,y,w):
    pm = np.zeros(len(X[0]))
    s1 = np.std(X,0)    
    s2 = np.std(y)
    for i in range(len(X[0])):
        cv = np.cov(X[:,i],y)
        print 'cv1',cv
        pm[i] = (cv[0][1])/s1[i]*s2
    print 'cv',cv[0][1]
    plt.scatter(pm,w)
    print 'pm',pm,'w',w
    print np.amax(pm)
    print np.amax(w)
    plt.xlabel('weight vector',fontsize=18)
    plt.ylabel('Pearson correlation co-efficient',fontsize=18)
    plt.show()
    