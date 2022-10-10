from random import seed
import numpy as np
import matplotlib.pyplot as plt
import sklearn

"""
Understanding how a perceptron works
"""

def make_dataset():
    N = 100
    Xdim0 = np.linspace(-2,16,N)
    Xdim1 = Xdim0*1.3+3
    Y = np.zeros((N))
    for i in range(N):
        if i<=N/2:
            Xdim1[i] += np.random.normal(loc=3,scale=0.4)
            Y[i] = -1
        else:
            Xdim1[i] -= np.random.normal(loc=6,scale=0.23)
            Y[i] = 1
    X = np.concatenate((Xdim0.reshape(-1,1),Xdim1.reshape(-1,1)),axis=1)
    return X,Y

def perceptron(X_seq: np.ndarray, Y_seq: np.ndarray, learning_rate=1e-3) -> np.ndarray:
    N = X_seq.shape[0]
    D = X_seq.shape[1]
    max_iterations = 6000 # in case not linearly sepearable
    W = np.zeros(D+1)
    X_seq = np.concatenate((X_seq, np.ones((N,1))), axis=1)
    for iters in range(max_iterations):
        mistakes = 0
        for k in range(N):
            yhat = np.sign(Y_seq[k]*np.dot(W,X_seq[k,:].reshape(-1)))
            if yhat<=0:
                W = W + learning_rate * Y_seq[k] * X_seq[k,:].reshape(-1)
                mistakes+=1
        if mistakes==0:
            break
    # print(f"Converged after {iters+1} epochs")
    return W

def main():
    X,Y = make_dataset()
    # Plot data
    Xdim0 = X[:,0]
    Xdim1 = X[:,1]
    plt.scatter(Xdim0,Xdim1,c=Y)
    # Set seed
    np.random.seed(0)
    indices = np.arange(X.shape[0])
    # Change the order data is provided, observe how it can affect the final hypothesis
    N_randoms = 5
    for r in range(N_randoms):
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        W = perceptron(X,Y)
        xplt = np.linspace(-2,16,1000)
        yplt = (-W[2]-W[0]*xplt)/W[1]
        plt.plot(xplt,yplt,label=f"iter{r}")
    plt.legend()
    plt.show()

if __name__=='__main__':
    main()