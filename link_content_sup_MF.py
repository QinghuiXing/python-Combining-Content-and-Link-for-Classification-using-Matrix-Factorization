import sklearn.decomposition.pca as pca
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import utils

def gradU(U,Z,Z2,A,gamma=0.01):
    U_grad=np.matmul(np.matmul(Z2,U),Z2)-\
        np.matmul(np.matmul(Z.T,A),Z)+gamma*U
    return U_grad

def gradV(V,Z,Z2,C,alpha=1,beta=0.01):
    V_grad=alpha*(np.matmul(V,Z2)-np.matmul(C.T,Z))+beta*V
    return V_grad

def gradZ(Z,A,V,C,G,W,ZU1,ZU2,alpha,lamda):
    Z_grad=(np.matmul(np.matmul(ZU1,Z.T),ZU2)+\
            np.matmul(np.matmul(ZU2,Z.T),ZU1)-\
            np.matmul(A.T,ZU2)-\
            np.matmul(A,ZU1))+\
            alpha*(np.matmul(np.matmul(Z,V.T),V)-\
                   np.matmul(C,V))+\
            lamda*np.matmul(G,W)
    return Z_grad

def gradW(G,Z,W,lamda,mu):
    W_grad=lamda*np.matmul(G.T,Z)+mu*W
    return W_grad

def gradb(G,lamda):
    n, _ = G.shape  # shape (n,c)
    I=np.ones(shape=(n,1))
    b_grad=lamda*np.matmul(G.T,I)
    return b_grad

def new_hinge_loss(x):
    if x>=2:
        return 0
    if x<=0:
        return 1-x
    x=(x-2)**2
    return 1/4*x

def error_for_sup(W,Z,Y,b,lamda,mu):
    _, YH = computeG(Y,Z,W,b)
    error = lamda*(np.sum(np.sum(YH)))+mu/2*(np.linalg.norm(W)**2)
    return error


def computeG(Y,Z,W,b):
    n,_ = Z.shape  # shape(n,l)
    I=np.ones(shape=(n,1))
    H=np.matmul(Z,W.T)+np.matmul(I,b.T)
    YH=np.multiply(Y,H)  # multi between corresponding position
    p,q=YH.shape
    for i in range(p):
        for j in range(q):
            YH[i,j]=new_hinge_loss(YH[i,j])
    return np.multiply(Y,YH),YH


def calculate_error(A,C,U,V,Z,alpha,beta,gamma,W,Y,b,lamda,mu):
    error = np.linalg.norm(A-np.matmul(np.matmul(Z,U),Z.T))**2+\
        alpha*(np.linalg.norm(C-np.matmul(Z,V.T))**2)+\
        gamma*(np.linalg.norm(U)**2)+\
        beta*(np.linalg.norm(V)**2)+\
        error_for_sup(W,Z,Y,b,lamda,mu)
    return error

def link_content_MF(A,C,Y,l,c,alpha=0.1,lamda=0.1,beta=0.01,gamma=0.01,mu=0.01,iter_num=1000,learning_rate=0.001):
    print("link & content Matrix Factorization...")
    assert type(A)==np.ndarray
    n,_=A.shape # shape (n,n)
    _,m=C.shape # shape (n,m)

    # initialization of parameters
    U=np.random.randn(l,l)/2000.0
    V=np.random.randn(m,l)/2000.0
    Z=np.random.randn(n,l)/2000.0
    W=np.random.randn(c,l)/2000.0
    b=np.random.randn(c,1)/2000.0

    # error recorder
    err_list = []

    # training
    for t in range(iter_num):
        error = calculate_error(A,C,U,V,Z,alpha,beta,gamma,W,Y,b,lamda,mu)
        err_list.append(error)
        if (error > 1000000000):
            print("exploded!!!")

        # share computation
        Z2=np.matmul(Z.T,Z)
        ZU1=np.matmul(Z,U.T)
        ZU2=np.matmul(Z,U)
        G,_=computeG(Y,Z,W,b)

        # calculate gradients
        U_grad=gradU(U,Z,Z2,A,gamma)
        V_grad=gradV(V,Z,Z2,C,alpha,beta)
        Z_grad=gradZ(Z,A,V,C,G,W,ZU1,ZU2,alpha,lamda)
        W_grad=gradW(G,Z,W,lamda,mu)
        b_grad=gradb(G,lamda)
        print(U_grad,V_grad,Z_grad,W_grad,b_grad)

        #update parameters
        U -= learning_rate*U_grad
        V -= learning_rate*V_grad
        Z -= learning_rate*Z_grad
        W -= learning_rate*W_grad
        b -= learning_rate*b_grad


    return Z,U,V,W,b,err_list


if __name__=='__main__':
    '''
    A =np.array([
    [0,1,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,1],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0,0,0,0,0,0,0,0]
    ])

    C,_ = datasets.make_classification(
        n_samples=8,n_features=10,
        n_classes=2,n_clusters_per_class=2
    )
    C=np.array(C)
    '''


    adj, features, labels, idx_train, idx_val, idx_test = utils.load_data()
    A=adj.A
    C=features.A
    _,c=labels.shape

    Z,U,V,W,b,err_list=link_content_MF(A,C,labels,l=50,c=c,alpha=1,lamda=1,beta=0.00001,gamma=0.00001,mu=0.00001,learning_rate=0.1,iter_num=100)

    print("Z(n x l) :\n",Z)
    print("U(l x l) :\n",U)
    print("V(m x l) :\n",V)
    print(err_list[-1])
    err_log=np.log(np.array(err_list))

    plt.plot(err_list)
    plt.show()
    plt.figure(2)
    plt.plot(err_log)
    plt.show()

