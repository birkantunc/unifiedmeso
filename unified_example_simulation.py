#!/usr/bin/python
#  @brief Example file to use networkInference class with simulated networks

#  Identifies communities and core-periphery structures
#  For details please refer to:
#  Birkan Tunc, Ragini Verma, Unifying Inference of Meso-Scale Structures in Networks, PLoS ONE 10(11): e0143133. doi:10.1371/journal.pone.0143133
#
#  @version 0.1.0
#  @date 2015/09/04 09:40
#
#  @author Birkan Tunc
#

import numpy as np
import sys
from os.path import realpath, join, dirname

sys.path.append(realpath(join(dirname(__file__), 'libYNA')))
from libYNA_network import networkInference

## @brief A function to calculate modularity measure for a given community structure
#
#  @param A is the connectivity (adjacency) matrix of the network
#  @param cIDs is the array including community ID of each node
def modularity(A, cIDs):
    #Total degree of the network
    M = A.sum()
    #Node degress
    k = np.squeeze(np.asarray(A.sum(axis=1)))

    Q = 0.
    for i in range(A.shape[0]-1):
        for j in range(i+1, A.shape[0]-1):
            if cIDs[i] != cIDs[j]: continue
            Q += A[i,j] - k[i]*k[j] / M

    return Q / (M/2.)

## @brief A function to simulate a random network with a specific meso-scale structure
#
#  @param N is the number of nodes per community
#  @param numG is the number of communities in the network
#  @param numC is the number of core nodes per community
#  @param coreType is the type of core-periphery structure (equations 7 and 8 in the paper)
def simulateNetwork(N, numG=1, numC=0, coreType=1):
    totalN = N * numG #total number of nodes in the network

    #Initialize network with random connections
    A = np.random.randint(0,5,[totalN,totalN])

    #If we want a core-periphery structure
    if numC > 0:
        realC = np.zeros(totalN)
    else:
        realC = np.ones(totalN)
        numC = N - numC

    for g in range(numG):                  #for each community
        if numC:
            realC[(N*g):(N*g+numC)] = 1    #identify core nodes

        for i in range(N-1):               #for each node
            ii = N*g + i
            for j in range(i+1, N):        #for each other node
                jj = N*g + j
                if (i<numC) and (j<numC):  #between two core nodes
                    A[ii,jj] = A[jj,ii] = np.random.randint(13,25)
                elif (i<numC) or (j<numC): #between one core and one periphery node
                    #see equation 7 and 8 in the paper for explanation of types
                    if coreType == 1:
                        A[ii,jj] = A[jj,ii] = np.random.randint(13,25)
                    elif coreType == 2:
                        A[ii,jj] = A[jj,ii] = np.random.randint(7,13)
                else:                      #between two periphery nodes
                    A[ii,jj] = A[jj,ii] = np.random.randint(0,5)
    return (A + A.T) / 2.


#In following examples, we assume to know the true parameters of the underlying structure
#such as true number of communities and core type. You can play with these numbers and
#use log-likelihood to predict the true values

##########################################################
#                    Example - I
##########################################################
#Simulate a network with community structure
A = simulateNetwork(N=12, numG=3, numC=0)
ni = networkInference(A)
#Random fit
randomLogLikelihood1 = ni.getRandomFit()
#Infer underlying structure
assignment1, logLikelihood1 = ni.solveCommunityModel(K=3, maxIt=5000)

##########################################################
#                    Example - II
##########################################################
#Simulate a network with core-periphery structure (1)
A = simulateNetwork(N=12, numG=1, numC=5, coreType=1)
ni = networkInference(A)
#Random fit
randomLogLikelihood2 = ni.getRandomFit()
#Infer underlying structure
coreness2, _, logLikelihood2 = ni.solveHybridModel(K=1, maxInIt=5000, coreType=1)

##########################################################
#                    Example - III
##########################################################
#Simulate a network with core-periphery structure (2)
A = simulateNetwork(N=12, numG=1, numC=5, coreType=2)
ni = networkInference(A)
#Random fit
randomLogLikelihood3 = ni.getRandomFit()
#Infer underlying structure
coreness3, _, logLikelihood3 = ni.solveHybridModel(K=1, maxInIt=5000, coreType=2)

##########################################################
#                    Example - IV
##########################################################
#Simulate a network with hybrid structure
A = simulateNetwork(N=12, numG=3, numC=5, coreType=1)
ni = networkInference(A)
#Random fit
randomLogLikelihood4 = ni.getRandomFit()
#Infer underlying structure
coreness4, assignment4, logLikelihood4 = ni.solveHybridModel(K=3, maxIt=100, maxInIt=5000, tol=1e-3, coreType=1)

#import matplotlib.pylab as plt
#A = simulateNetwork(12,3,5,1)
#plt.imshow(A)
#plt.colorbar()
#plt.axis('off')
#plt.show()
