#混合ガウス ギブスサンプリング

import numpy as np
import math
def dataset(n,n_cluster):

    true_mu = []
    true_mu.append([2,10])
    true_mu.append([-17,0])
    true_mu.append([8,-7])
    true_mu.append([20,0])
    
    true_cov = []

    true_cov.append([[3,1],[1,12]])
    true_cov.append([[11,2],[2,9]])
    true_cov.append([[5,0],[0,18]])
    true_cov.append([[5,5],[5,1]])

    data = []
    for k in range(n_cluster):
        data.append(np.random.multivariate_normal(true_mu[k],true_cov[k],int(n/n_cluster)))

    data = np.vstack(data)

    return data

def sampling_S(N,K,pi,mu,lam,X):
    eta = np.zeros((N,K))
    S = np.zeros((N,K))
    for n in range(N):
        for k in range(K):
            x_m = X[n]-mu[k]
            eta[n,k] = np.exp(-0.5*x_m.dot(lam[k]).dot(x_m.T) + np.log(pi[k])+0.5*np.log(np.linalg.det(lam[k])))
  
    eta = eta/(eta.sum(axis=1).reshape(-1,1))
    for n in range(N):
        S[n] = np.random.multinomial(1,eta[n])
    
    return S

def sampling_pi(N,K,S):
    alpha = np.ones(K)
    alpha_hat = np.zeros(K)
    
    alpha_hat = S.sum(axis = 0) + alpha
    pi = np.random.dirichlet(alpha_hat)
    return pi

def sampling_mu_lam(N,K,X,S,D):
    m = np.ones(D)
    beta = 1.0
    v = np.ones(K)*2
    W = np.eye(2)

    beta_hat = np.zeros(K)
    m_hat = np.zeros((K,2))
    
    pred_mu = np.ones((K,D))
    pred_lam = np.array([np.eye(D) for i in range(K)])
    for k in range(K):
        tm = np.zeros(D)
        for n in range(N):
            tm += S[n,k]*X[n]
            
        beta_hat[k] = S[:,k].sum() + beta
        m_hat[k] = (tm + beta*m)/beta_hat[k]

    for k in range(K):
        tm = np.zeros([D,D])
        for n in range(N):
            tm += S[n,k]*(X[n].reshape(-1,1).dot(X[n].reshape(-1,1).T))
        

        v_hat = S[:,k].sum() + v[k]
        W_hat_inv = tm + beta*m.T.dot(m) - beta_hat[k]*m_hat[k].reshape(-1,1).dot(m_hat[k].reshape(-1,1).T)+ np.linalg.inv(W)

        pred_lam[k]= v_hat * np.linalg.inv(W_hat_inv) # wishart分布からサンプリングできなかったので、期待値
        pred_mu[k] = np.random.multivariate_normal(m_hat[k], np.linalg.inv(beta_hat[k]*pred_lam[k]))
        
    return pred_mu,pred_lam


K = 4
D=2
N = 400
X = dataset(N,K)

pi = np.ones(K)

S = np.ones([N,K])
mu=np.ones((K,D))
lam = np.array([np.eye(D) for i in range(K)])

max_iter = 100

for i in range(max_iter):
    
    S = sampling_S(N,K,pi,mu,lam,X)

    mu,lam = sampling_mu_lam(N,K,X,S,D)

    pi = sampling_pi(N,K,S)

    
c = ["r","g","b","y"]
fig = plt.figure(figsize = (15,5))

plt.subplot(1,3,1)
plt.scatter(X[:,0],X[:,1])
plt.title("train data")

plt.subplot(1,3,2)
for i in range(N):
    plt.scatter(X[i,0],X[i,1],c=c[list(S[i]).index(1)])
    
plt.title("putting color by latent variable")

plt.subplot(1,3,3)

for k in range(K):
    x = np.random.multivariate_normal(mu[k],np.linalg.inv(lam[k]),size=100)
    plt.scatter(x[:,0],x[:,1],c=c[k])
plt.title("sampling by predicted parameter")
