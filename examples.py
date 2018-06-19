# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:54:20 2017

@author: tuncb
"""
import numpy as np
import matplotlib.pylab as plt

from meso import core_periphery, communities, communities_of_core_periphery

#######################################################################
# Normalized mutual information to measure quality of solutions
#######################################################################
def NMI(cIDs1, cIDs2):
    numC1 = len(np.unique(cIDs1))
    numC2 = len(np.unique(cIDs2))
    N = float(len(cIDs1))

    numerator = 0.
    denominator1 = 0.
    denominator2 = 0.
    for c1 in range(numC1):
        N1 = float(np.sum(cIDs1==c1))
        term = N1 * np.log(N1/N) if N1!=0 else 0.
        denominator1 += term

        for c2 in range(numC2):
            N2 = float(np.sum(cIDs2==c2))
            if c1 == 0:
                term = N2 * np.log(N2/N) if N2!=0 else 0.
                denominator2 += term


            N12 = np.sum((cIDs1==c1)&(cIDs2==c2))
            term = N12 * np.log((N12*N) / (N1*N2)) if N12!=0 else 0.
            numerator += term

    nmi = -2 * numerator / (denominator1 + denominator2)
    return nmi

#######################################################################
# Simulate data having both communities and core-periphery structures
#######################################################################
#Generate an adjacency matrix (node-to-node similarity)
N = 100               #Total number of nodes
numC = 40             #Number of core nodes
numP = N - numC       #Number of periphery nodes
K = 4                 #Number of communities

A = np.random.randint(0,3,[N,N])
realC = np.zeros(N, dtype='int')
realC[0:numC] = 1
groups = np.random.randint(0,K,N)
np.random.shuffle(realC)
np.random.shuffle(groups)

for i in range(N-1):
    for j in range(i+1, N):
        core = (realC[i] + realC[j] - realC[i]*realC[j])
        extra = (groups[i] == groups[j]) * core
        A[i,j] += extra * np.random.randint(0,6) #extra connections that nodes made based on their communities and coreness
        A[j,i] = A[i,j]
np.fill_diagonal(A, 0)

#clean some
clean = int(N  * 0.3) #amount of cleaning
for i in range(N):
    idx = np.random.randint(0,N,clean)
    A[i, idx] = 0
    A[idx, i] = 0
idx = np.where(A<2)
order = np.arange(len(idx[0]))
np.random.shuffle(order)
remove = int(len(idx[0]) * 0.1)
idx = (idx[0][order[0:remove]], idx[1][order[0:remove]])
A[idx] = 0

#display the adjacency matrix
order = np.lexsort((realC,groups))
plt.imshow(A[order,:][:,order], cmap='jet', interpolation='none')
plt.colorbar()
plt.axis('off')
plt.show()

#######################################################################
# infer latent configuration (communities, cores etc.)
#######################################################################
#Inputs:
#A: adjacency matrix
#K: number of communities
#fullOptimization: whether optimize scale parameter as weel; takes longer but results in much better solutions
#binMask: whether discard zero entries of A when making optimization; usually results in better solutions
 
#Outcomes:
#logs: loglikelihood values for three scenarios
#      (1) random network, preserving node degrees
#      (2) inferred configuration
#      (3) optimal configuration for the generative model (expectations = observations)
#      Scenario (3) can be considered as the maximum value under the assumption that generative model is true
#Z: community assignments
#c: coreness of nodes (1: core, 0: periphery)
#wM: inferred scale paramater (how much being in the same cummunity/core adds to expected connections between nodes)

#searching for only community structure
logs, Z0, wM = communities(A, K=K, fullOptimization=True, binMask=True)

#searching for only core-periphery structure
logs, c0, wM = core_periphery(A, fullOptimization=True, binMask=True)

#searching for communities of core-periphery structures (i.e. hybrid)
logs, Z1, c1, wM = communities_of_core_periphery(A, K=K, fullOptimization=True, binMask=True)

#reporting agreement between actual and predicted configurations
print (c0==realC).sum() / float(len(c0)), NMI(Z0, groups)
print (c1==realC).sum() / float(len(c1)), NMI(Z1, groups)

