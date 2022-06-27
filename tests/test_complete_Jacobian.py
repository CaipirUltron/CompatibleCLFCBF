import numpy as np
from numpy import random as rd

def Projection(v,w):
    return ( np.eye(n) - np.outer(v,w)/(v.T @ w) )

n = 2
m = 3

# Parameters
p = rd.rand()
alpha = rd.rand()
gamma = alpha

# Equilibrium conditions
V = rd.rand()
h = 0.0

gammaV = gamma*V
alphah = alpha*h

# Gradients
nablaV = rd.rand(n)
nablah = rd.rand(n)

g = rd.rand(2,3)
G = g @ g.T

normV = nablaV.T @ G @ nablaV
normh = nablah.T @ G @ nablah

GnablaV = G @ nablaV
Gnablah = G @ nablah

P_GnablaV = Projection( GnablaV, nablaV )
P_Gnablah = Projection( Gnablah, nablah )
eta = 1/( 1 + p * (nablaV.T @ P_Gnablah @ GnablaV) )

lambda1 = p*gammaV
lambda2 = rd.rand()

rootHv = rd.rand(n,n)
Hv = rootHv.T @ rootHv

rootHh = rd.rand(n,n)
Hh = rootHh.T @ rootHh

H = lambda2*Hh - lambda1*Hv
w = lambda2*nablah - lambda1*nablaV

f = - G @ w
print("f + Gw = " + str(f + G @ w))

Jf = rd.rand(n,n)

delG = np.zeros([n,n])
for i in range(n):
    delG_i = rd.rand(n,n)
    delG[:,i] = delG_i @ w

Hs = G @ H + delG + Jf

outerV = np.outer( GnablaV, nablaV )
outerh = np.outer( Gnablah, nablah )

Jcl = ( np.eye(n) - eta*( outerh/normh + p * P_Gnablah @ outerV @ P_Gnablah ) ) @ Hs
- eta * ( alpha * ( np.eye(n) + p*normV*P_GnablaV ) @ ( outerh/normh ) + p * gamma * P_Gnablah @ outerV )

z1 = nablah
test1 = z1.T @ Jcl - z1.T @ ( (1-eta)*Hs - alpha*np.eye(n) )

print(z1)