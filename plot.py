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