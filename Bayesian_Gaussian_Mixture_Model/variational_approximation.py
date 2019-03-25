#混合ガウス変分推論

import numpy as np
import math
from scipy.special import digamma
def dataset(n,n_cluster):
    np.random.seed(1103)
    
    true_mu = []
    true_mu.append([2,10])
    true_mu.append([-10,0])
    true_mu.append([18,-7])
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

class Variational_Inference:
    
    def __init__(self,N,K,D,max_iter):
        self.K = K
        self.D=D
        self.N =N 
        self.pi = np.ones(K)
        self.S = np.ones([N,K])
        self.mu=np.ones((K,D))
        self.lam = np.array([np.eye(D) for i in range(K)])
    
        self.v_hat = np.ones(K)*2
        self.W_hat = np.array([np.eye(D) for i in range(K)])
        self.m_hat = np.zeros((K,D))
        self.beta_hat = np.ones(K)
        self.alpha_hat = np.ones(K)
        
        self.max_iter = max_iter
    
    def logsumexp(self,X):
        # \log(\sum_{i=1}^{N}\exp(x_i))
        max_x = np.max(X, axis=1).reshape(-1, 1)
        return np.log(np.sum(np.exp(X - max_x), axis=1).reshape(-1, 1)) + max_x

    def update_S(self,pi, mu,lam,X):
        eta = np.zeros((self.N,self.K))
        S = np.zeros((self.N,self.K))
        
        #期待値計算
        E_log_pi = digamma(self.alpha_hat) - digamma(self.alpha_hat.sum())
        E_lam = np.array([np.eye(self.D) for i in range(self.K)])
        E_lam_mu = np.ones([self.K,self.D])
        E_mu_lam_mu = np.ones((self.K,1))
        E_log_pi = digamma(self.alpha_hat) - digamma(self.alpha_hat.sum())
        
        for n in range(self.N):
            for k in range(self.K):
                tmp=0
                E_lam[k] = self.v_hat[k]*self.W_hat[k]
                
                E_lam_mu[k] = self.v_hat[k]*self.W_hat[k].dot(self.m_hat[k].T)
                E_mu_lam_mu[k] = (self.v_hat[k]*self.m_hat[k].dot(self.W_hat[k]).reshape(1,-1).dot(self.m_hat[k]).reshape(-1,1)+self.D/self.beta_hat[k])[0][0]
                for d in range(self.D):
                    tmp += digamma((self.v_hat[k]+1-d)/2)
                E_log_lam = tmp + self.D*np.log(2) + np.log(np.linalg.det(self.W_hat[k]))
                x_m = X[n]-mu[k]
                eta[n,k] = -0.5*x_m@E_lam[k]@x_m + X[n]@E_lam_mu[k] - 0.5*E_mu_lam_mu[k] +0.5*E_log_pi[k]+0.5*E_log_lam
                

        #eta = eta/(eta.sum(axis=1).reshape(-1,1))
        eta = np.exp(eta-self.logsumexp(eta))
        for n in range(N):
            S[n] = np.random.multinomial(1,eta[n])

        return S,eta

    def update_pi(self,S):
        alpha = np.ones(self.K)
        self.alpha_hat = S.sum(axis = 0) + alpha
        pi = np.random.dirichlet(self.alpha_hat)
        return pi

    def update_mu_lam(self,X,S,eta):
        #事前分布のパラメータ
        m = np.ones(self.D)
        v = 3
        W = np.eye(2)
        beta = 2
        pred_mu = np.ones((self.K,self.D))
        pred_lam = np.array([np.eye(self.D) for i in range(self.K)])
        for k in range(self.K):
            tm = np.zeros(self.D)
            for n in range(self.N):
                tm += eta[n,k]*X[n]

            self.beta_hat[k] = eta[:,k].sum() + beta
            self.m_hat[k] = (tm + beta*m)/self.beta_hat[k]

        for k in range(self.K):
            tm = np.zeros([self.D,self.D])
            for n in range(N):
                tm += eta[n,k]*(X[n].reshape(-1,1).dot(X[n].reshape(-1,1).T))

            
            self.v_hat[k] = eta[:,k].sum() + v
            W_hat_inv = tm + beta*m.T.dot(m) - self.beta_hat[k]*self.m_hat[k].reshape(-1,1).dot(self.m_hat[k].reshape(-1,1).T)+ np.linalg.inv(W)

            pred_lam[k]= self.v_hat[k] * np.linalg.inv(W_hat_inv) # wishart分布からサンプリングできなかったので、期待値
            pred_mu[k] = np.random.multivariate_normal(self.m_hat[k], np.linalg.inv(self.beta_hat[k]*pred_lam[k]))
            

        return pred_mu,pred_lam

K = 4
D=2
N = 200
X = dataset(N,K)

pi = np.ones(K)
S = np.ones([N,K])
mu=np.ones((K,D))
lam = np.array([np.eye(D) for i in range(K)])

max_iter = 20

model = Variational_Inference(N,K,D,max_iter)

for i in range(max_iter):
    
    S,eta = model.update_S(pi,mu,lam,X)

    mu,lam = model.update_mu_lam(X,S,eta)

    pi = model.update_pi(S)

import matplotlib.pyplot as plt
c = ["r","g","b","y"]
fig = plt.figure(figsize = (15,5))

plt.subplot(1,3,1)
plt.scatter(X[:,0],X[:,1])
plt.title("train data")

plt.subplot(1,3,2)
for i in range(N):
    plt.scatter(X[i,0],X[i,1],c=c[list(S[i]).index(1)])

plt.title("put color by latent variable")

plt.subplot(1,3,3)

for k in range(K):
    x = np.random.multivariate_normal(mu[k],np.linalg.inv(lam[k]),size=100)
    plt.scatter(x[:,0],x[:,1],c=c[k])
plt.title("sampling by predicted parameter")

