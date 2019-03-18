#崩壊型ギブスサンプリング

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom
import math

np.random.seed(1103)

def logsumexp(X):
    # \log(\sum_{i=1}^{N}\exp(x_i))
    max_x = np.max(X, axis=1).reshape(-1, 1)
    return np.log(np.sum(np.exp(X - max_x), axis=1).reshape(-1, 1)) + max_x


def cmb(n, r):
    if n - r < r: r = n - r
    if r == 0: return 1
    if r == 1: return n

    numerator = [n - r + k + 1 for k in range(r)]
    denominator = [k + 1 for k in range(r)]

    for p in range(2,r+1):
        pivot = denominator[p - 1]
        if pivot > 1:
            offset = (n - r) % p
            for k in range(p-1,r,p):
                numerator[k - offset] /= pivot
                denominator[k] /= pivot

    result = 1
    for k in range(r):
        if numerator[k] > 1:
            result *= int(numerator[k])

    return result

N = 400
#train data
x1 = np.random.poisson(lam = 100,size = 150)
x2 = np.random.poisson(lam = 10, size = 250)
X = np.r_[x1,x2].reshape(-1,1)

#number of class
K = 2

#latent var
S = np.ones([N,K])

#parameters of gamma dist
a=1
b=1
a_hat = X.T.dot(S) + a
b_hat = S.sum(axis=0)+ b

#parameters of dirichlet
alpha = np.random.randn(1,K)
alpha_hat = S.sum(axis=0)+ alpha


max_iter = 10

for i in range(max_iter):
    for n in range(N):

        a_hat -= S[n]*X[n]
        b_hat -= S[n]
        alpha_hat -= S[n]
        
        gam = a_hat/(a_hat).sum(axis=1)
        bb = 1./(b_hat+1.)
        log_cnb = np.array([math.log(cmb(int(X[n][0]+i-1),int(X[n][0]))) for i in a_hat[0]])
        ex = np.log(gam) + a_hat*np.log(1-bb) + X[n]*np.log(bb) + log_cnb
        ex = np.exp(ex-logsumexp(ex))
        S[n] = np.random.multinomial(1,ex[0],1)
        
        a_hat += S[n]*X[n]
        b_hat += S[n]
        alpha_hat += S[n]


fig = plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.hist(x1,alpha=0.7)
plt.hist(x2,alpha=0.7)
plt.title("True")

plt.subplot(1,2,2)
c = ["r","g"]
x,y=[],[]
for i in range(N):
    if int(np.array(S)[:,0][i]) == 0:
        x.append(X[i])
    else:
        y.append(X[i])

plt.hist(np.array(x).flatten(),alpha=0.7)
plt.hist(np.array(y).flatten(),alpha=0.7)
plt.title("sampling")
plt.show()
