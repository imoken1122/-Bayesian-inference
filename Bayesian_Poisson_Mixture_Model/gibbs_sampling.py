import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1103)

def softmax(X):
    l = []
    for x in X:
        sum_ex = np.sum(np.exp(x))
        l.append([np.exp(i)/sum_ex for i in x])
    return np.array(l)

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
a = np.random.randn(1,K)
b = np.random.randn(1,K)

#parameters of dirichlet
alpha = np.random.randn(1,K)

#train data
x1 = np.random.poisson(lam = 20,size = 150)
x2 = np.random.poisson(lam = 10, size = 250)
X = np.r_[x1,x2].reshape(-1,1)

max_iter = 100

S_sample,lam_sample,pi_sample = [],[],[]

for i in range(max_iter):
    
    #潜在変数
    ex = X.dot(np.log(lam)) - lam + np.log(pi)
    ex = softmax(ex)

    for n in range(N):
        S[n] = np.random.multinomial(1,ex[n],1)
    S_sample.append(S)
    
    #λ
    a_hat = X.T.dot(S) + a    
    b_hat = S.sum(axis = 0) + b

    lam = np.random.gamma(a_hat,1/b_hat)
    lam_sample.append(lam.flatten())
    
    #π
    alpha_hat= S.sum(axis = 0) + alpha[0]
    pi = np.random.dirichlet(alpha_hat)
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
