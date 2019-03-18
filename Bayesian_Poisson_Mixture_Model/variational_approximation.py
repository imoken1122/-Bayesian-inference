import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma
from scipy.stats import dirichlet,gamma
np.random.seed(1103)
def logsumexp(X):
    max_x = np.max(X, axis=1).reshape(-1, 1)
    return np.log(np.sum(np.exp(X - max_x), axis=1).reshape(-1, 1)) + max_x

N = 400

#number of class
K = 2

#paramter of poisson dist
lam = np.ones([1,K]).reshape(1,-1)

#parameter of category dist
pi = np.array([0.5,0.5]).reshape(1,-1)

#latent var
S = np.zeros([N,K])

#parameters of gamma dist
a = 1
b = 1
a_hat=np.ones([1,K])
b_hat=np.ones([1,K])
#parameters of dirichlet
alpha = np.random.randn(1,K)
alpha_hat= np.random.randn(1,K)
#train data
x1 = np.random.poisson(lam = 20,size = 150)
x2 = np.random.poisson(lam = 10, size = 250)
X = np.r_[x1,x2].reshape(-1,1)

max_iter = 1000
sums = 0
lam_sample,pi_sample=[],[]
for i in range(max_iter):
    
    E_lam = a_hat/b_hat
    E_log_lam = digamma(a_hat) - np.log(b_hat)
    E_log_pi = digamma(alpha_hat) - digamma(alpha_hat.sum())

    #潜在変数
    ex = X.dot(E_log_lam) - E_lam + E_log_pi
    tmp = -logsumexp(ex)
    ex = np.exp(ex+tmp)
    
    #λ
    a_hat = X.T.dot(ex) + a 
    b_hat = ex.sum(axis = 0) + b
    lam = np.random.gamma(a_hat,1/b_hat)
    lam_sample.append(lam.flatten())
    
    #π
    alpha_hat= ex.sum(axis = 0) + alpha
    pi = np.random.dirichlet(alpha_hat[0])
    pi_sample.append(pi.flatten())


pi_sample=np.array(pi_sample)
lam_sample=np.array(lam_sample)

pi1_mean = pi_sample[:,0].mean()
pi2_mean = pi_sample[:,1].mean()
lam1_mean = lam_sample[:,0].mean()
lam2_mean = lam_sample[:,1].mean()

fig = plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.hist(pi_sample[:,0],bins=10,alpha=0.6,label = f"{pi1_mean}")
plt.hist(pi_sample[:,1],bins=10,alpha=0.6,label=f"{pi2_mean}")
plt.xlabel("$π$")
plt.legend()

plt.subplot(1,2,2)
plt.hist(lam_sample[:,0],bins=10,alpha=0.6,label = f"{lam1_mean}")
plt.hist(lam_sample[:,1],bins=10,alpha=0.6,label=f"{lam2_mean}")
plt.xlabel("$λ$")
plt.legend()
plt.show()
