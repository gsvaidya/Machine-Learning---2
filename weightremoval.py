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