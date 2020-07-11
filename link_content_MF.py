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

def gradZ(Z,A,V,C,ZU1,ZU2,alpha):
    Z_grad=(np.matmul(np.matmul(ZU1,Z.T),ZU2)+\
            np.matmul(np.matmul(ZU2,Z.T),ZU1)-\
            np.matmul(A.T,ZU2)-\
            np.matmul(A,ZU1))+\
            alpha*(np.matmul(np.matmul(Z,V.T),V)-\
                   np.matmul(C,V))
    return Z_grad

def calculate_error(A,C,U,V,Z,alpha,beta,gamma):
    error = np.linalg.norm(A-np.matmul(np.matmul(Z,U),Z.T))**2+\
        alpha*(np.linalg.norm(C-np.matmul(Z,V.T))**2)+\
        gamma*(np.linalg.norm(U)**2)+\
        beta*(np.linalg.norm(V)**2)
    return error

def link_content_MF(A,C,l,alpha=0.1,beta=0.01,gamma=0.01,iter_num=1000,learning_rate=0.001):
    print("link & content Matrix Factorization...")
    assert type(A)==np.ndarray
    n,_=A.shape
    _,m=C.shape

    # initialization of parameters
    U=np.random.randn(l,l)/98.0
    V=np.random.randn(m,l)/98.0
    Z=np.random.randn(n,l)/98.0

    # error recorder
    err_list = []

    # training
    for t in range(iter_num):
        error = calculate_error(A,C,U,V,Z,alpha,beta,gamma)
        if (error > 1000000000):
            print("exploded!!!")
        err_list.append(error)

        # share computation
        Z2=np.matmul(Z.T,Z)
        ZU1=np.matmul(Z,U.T)
        ZU2=np.matmul(Z,U)

        # calculate gradients
        U_grad=gradU(U,Z,Z2,A,gamma)
        V_grad=gradV(V,Z,Z2,C,alpha,beta)
        Z_grad=gradZ(Z,A,V,C,ZU1,ZU2,alpha)

        # update parameters
        U -= learning_rate*U_grad
        V -= learning_rate*V_grad
        Z -= learning_rate*Z_grad

    return Z,U,V,err_list


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


    adj, features, _, _, _, _ = utils.load_data()
    A=adj.A
    C=features.A

    Z,U,V,err_list=link_content_MF(A,C,50,alpha=1,beta=0.001,gamma=0.001,learning_rate=0.1,iter_num=1000)

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

