#!/usr/bin/python
## @package YNA

#  @brief A basic class to infer meso-scale structures in networks

#  This class provides an interface to identify communities and core-periphery structures
#  For details please refer to:
#  Birkan Tunc, Ragini Verma, Unifying Inference of Meso-Scale Structures in Networks, PLoS ONE 10(11): e0143133. doi:10.1371/journal.pone.0143133

#  @version 0.1.0
#  @date 2015/09/03 15:42
#
#  @author Birkan Tunc
#

import numpy as np
import scipy as sp
from scipy.special import gammaln
from sklearn.cluster import spectral_clustering

## @brief Efficient and reliable summation for log values
#
#  @param array is the array including values to be summed
def logsum(array):
    amax = np.max(array, axis=0);
    rep = [1] * len(array.shape);
    rep[0] = array.shape[0];
    s = np.log(np.sum(np.exp(array - np.tile(amax, rep)), axis=0)) + amax;

    return s


class networkInference:
    ## @brief Constructor
    #
    #  Initialize a network form the connectivity (adjacency) matrix provided
    #
    #  @param A is the connectiviyt matrix including number of connections/interactions between nodes (rows and columns)
    def __init__(self, A):
        self.A = A*1.
        self.N = A.shape[0] #number of nodes in the network
        self.coreType = 1   #see equation 7 and 8 in the paper for explanation of types

    ## @brief Generative model for core-periphery structures. Defines expected number of interactions between nodes
    #
    #  @param c is the array of coreness values of nodes
    #  @param Z is the matrix with rows being nodes and columns being communities. Encodes probability of a node being part of a community.
    #  Parameter Z is meaningful only when hybrid models (communities of core-periphery structures) are considered
    def generativeCores(self, c, Z):
        N = self.N        #number of nodes in the network
        K = Z.shape[1]    #number of communities

        #Beta matrix encodes expected number of interactions between nodes
        #In hybrid models, each layer (K) has the information for a specific community
        B = np.zeros([K, N, N], dtype='float')
        for i in range(N-1):
            for j in range(i+1, N):
                for k in range(K):
                    if self.coreType == 1:
                        B[k,i,j] = B[k,j,i] = 1*(Z[i,k]*Z[j,k]) * (c[i] + c[j]- c[i]*c[j])
                    elif self.coreType == 2:
                        B[k,i,j] = B[k,j,i] = 1*(Z[i,k]*Z[j,k]) * (c[i] + c[j])
        return B

    ## @brief Generative model for community structures. Defines expected number of interactions between nodes
    #
    #  @param Z is the matrix with rows being nodes and columns being communities. Encodes probability of a node being part of a community.
    def generativeCommunities(self, Z):
        N = self.N        #number of nodes in the network
        K = Z.shape[1]    #number of communities


        #Beta matrix encodes expected number of interactions between nodes
        B = np.zeros([K, N, N], dtype='float')
        for i in range(N-1):
            for j in range(i+1, N):
                for k in range(K):
                    #We expect interactions between nodes of the same communities
                    #But not of different communities
                    B[k,i,j] = B[k,j,i] = (Z[i,k]*Z[j,k])
        return B

    ## @brief Generative model for random networks. Defines expected number of interactions between nodes
    def generativeRandom(self):
        N = self.N        #number of nodes in the network
        K = 1             #number of communities

        #Beta matrix encodes expected number of interactions between nodes
        B = np.zeros([K, N, N], dtype='float')
        for i in range(N-1):
            for j in range(i+1, N):
                for k in range(K):
                    #Interactions are expected betwen all pairs of nodes
                    B[k,i,j] = B[k,j,i] = 1.
        return B


    ## @brief Log-likelihood of observations, given expected number of interactions (B)
    #
    #  @param B is the matrix encoding expected number of interactions between nodes
    #  @param Z is the matrix with rows being nodes and columns being communities. Encodes probability of a node being part of a community.
    def likelihood(self, B, Z=None):
        A = self.A        #connectivity matrix of the network
        N = self.N        #number of nodes in the network

        if Z is not None: #if there are communities
            K = Z.shape[1]
            pr = Z.sum(axis=0)
            pr = pr / float(pr.sum()) #prior probabilities of communities
        else:            #if not
            K = 1
            pr = sp.array([1])


        #This part calculates the Dirichlet-Multinomial log-likelihood (see equation 3 of the paper)
        a = A.sum(axis=1)

        afl = gammaln(a+1)
        Afl = gammaln(A+1).sum(axis=1)

        bgl = np.zeros([N, K], dtype='float')
        abgl = np.zeros([N, K], dtype='float')
        ABgl = np.zeros([N, K], dtype='float')
        for k in range(K):
            Bk = B[k,:,:].copy()
            b = Bk.sum(axis=1)
            tmp = gammaln(b)
            tmp[b<1e-1] = 0.
            bgl[:,k] = tmp

            AB = A+Bk
            ab = AB.sum(axis=1)
            tmp = gammaln(ab)
            tmp[np.isnan(tmp)] = 0.
            tmp[ab<1e-1] = 0.
            abgl[:,k] = tmp

            tmp = gammaln(AB) - gammaln(Bk)
            tmp[np.isnan(tmp)] = 0.
            tmp[tmp<1e-12] = 0.
            ABgl[:,k] = (tmp).sum(axis=1)

        #log-likelihood of individual nodes for each community
        logL = np.tile(afl,[K,1]).T - np.tile(Afl,[K,1]).T + bgl - abgl + ABgl + np.log(pr.reshape([1, K]))

        #Joint log-likelihood of observations and communities
        #see section "Inferring Meso-Scale Structures" for details
        if Z is not None:
            norm = logsum(logL.T) #summation over communities. usefull in hybrid models
            jlogL = np.sum(Z * logL)
        else:
            norm = logL.flatten()
            jlogL = np.sum(logL) #summation of individual log-likelihoods

        return norm, logL, jlogL

    ## @brief Log-likelihood of observations for a random network
    def getRandomFit(self):
        B = self.generativeRandom()
        _, _, logL = self.likelihood(B)

        return logL

    ## @brief Random update for coreness values
    #
    #  @param c0 is the array of current coreness values
    def updaterCores(self, c0):
        ind = np.random.randint(0, len(c0))
        c1 = c0.copy()
        c1[ind] = 1. - c1[ind]

        #When using a non-binary model (coreness ranges between 0-1) use below two lines
        #update = np.random.normal(0, 0.1)
        #c1[ind] = min(1., max(0., c1[ind] + update))

        return c1

    ## @brief Random update for community assignments
    #
    #  @param Z0 is the matrix of current community assignments
    def updaterCommunities(self, Z0):
        K = Z0.shape[1]
        indN = np.random.randint(0, Z0.shape[0])
        indK = np.random.randint(0, K)
        Z1 = Z0.copy()
        Z1[indN, :] = 0.
        Z1[indN, indK] = 1.

        return Z1

    ## @brief Initialize coreness values
    def initializeCores(self):
        N = self.N
        c = np.zeros(N)

        return c

    ## @brief Initialize community assignments
    #
    #  @param random is the switch for random initilization. If False, spectral clustering is used
    def initializeCommunities(self, random=False):
        N = self.N        #number of nodes in the network
        K = self.K        #number of communities

        if K==1:
            Z = np.ones([N,1], dtype='float')
        else:
            Z = np.zeros([N,K])
            if random is False:
                labels = spectral_clustering(self.A, n_clusters=K, eigen_solver='arpack')
                Z[range(N), labels] = 1.
            else:
                for n in range(N):
                    idx = np.random.randint(0, K)
                    Z[n, idx] = 1
        return Z

    ## @brief Optimization for coreness values
    #
    #  @param Z is the matrix with rows being nodes and columns being communities. Encodes probability of a node being part of a community.
    #  @param c0 is the array of initial coreness values
    #  @param maxIt is the maximum number of iterations allowed
    def maximizeCores(self, c0, Z, maxIt=5000):
        #get the initial expected number of interactions
        B = self.generativeCores(c0, Z)
        #Joint likelihood of initial observations and assignment
        #If this is a pure core-periphery structure without any communities, Z is an array of ones
        norm, logL, f0 = self.likelihood(B, Z)

        counter = 0
        for it in range(maxIt):
            c1 = self.updaterCores(c0)
            B = self.generativeCores(c1, Z)
            _norm, _logL, f1 = self.likelihood(B, Z)

            #This is a very primitive deterministic optimization
            #Consider using a better method such as Simulated Annealing
            #by using a random process to calculate 'test'
            test = f1 > f0
            counter += 1

            if test:
                c0 = c1
                f0 = f1
                norm, logL = _norm, _logL
                counter = 0
                print("It: %0.4d S: %0.4d logL: %0.2f"%(it, counter, f0))
            elif counter > 750:
                print("It: %0.4d S: %0.4d logL: %0.2f"%(it, counter, f0))
                break

        return norm, logL, f0, c0

    ## @brief Optimization for community assignments
    #
    #  @param Z0 is the matrix of initial community assignments
    #  @param maxIt is the maximum number of iterations allowed
    def maximizeCommunities(self, Z0, maxIt=5000):
        #get the initial expected number of interactions
        B = self.generativeCommunities(Z0)
        #Joint likelihood of initial observations and assignment
        _norm, _logL, f0 = self.likelihood(B, Z0)

        counter = 0
        for it in range(maxIt):
            Z1 = self.updaterCommunities(Z0)
            B = self.generativeCommunities(Z1)
            _norm, _logL, f1 = self.likelihood(B, Z1)

            #This is a very primitive deterministic optimization
            #Consider using a better method such as Simulated Annealing
            #by using a random process to calculate 'test'
            test = f1 > f0
            counter += 1

            if test:
                Z0 = Z1
                f0 = f1
                print("It: %0.4d S: %0.4d logL: %0.2f"%(it, counter, f0))
                counter = 0
            elif counter > 750:
                print("It: %0.4d S: %0.4d logL: %0.2f"%(it, counter, f0))
                break

        return f0, Z0

    ## @brief Main solver for hybrid models
    #
    #  When the number of communities is 1, this solves for a pure core-periphery structure
    #
    #  @param K is the number of communities
    #  @param numRep is the number of repetation for the solver. Takes the best result
    #  @param maxIt is the maximum number of iterations allowed for EM algorithm
    #  @param maxInIt is the maximum number of iterations allowed for M phase (optimizing coreness values)
    #  @param tol is the stopping criterion
    #  @param coreType is the type of core-periphery structure (equations 7 and 8 in the paper)
    def solveHybridModel(self, K, numRep=10, maxIt=100, maxInIt=5000, tol=1e-6, coreType=1):
        self.K = K
        self.coreType = coreType

        bestC = 0.
        bestZ = 0.
        bestLogL = -1e+10

        #Make repeatation for better inference
        for rep in range(numRep):
            #Decide starting point
            c = self.initializeCores()
            Z = self.initializeCommunities()

            #Solve mixture model using EM algorithm
            f0 = 0.0
            if K==1: maxIt = 1
            for it in range(maxIt):
                #For a fixed Z optimize model parameters i.e. coreness vector
                norm, logL, f1, c = self.maximizeCores(c, Z, maxInIt)
                #Then update Z
                if K!=1:
                    Z = np.exp(logL - np.tile(norm,[K,1]).T)

                diff = np.abs(f1-f0)
                f0 = f1

                if diff <= tol:
                    break

            if f0 > bestLogL:
                bestC = c
                bestZ = Z
                bestLogL = f0

            print("Repeat:%d Best log-likelihood: %0.2f"%(rep, bestLogL))
            print('')

        return bestC, bestZ, bestLogL


    ## @brief Main solver for communities
    #
    #  @param K is the number of communities
    #  @param numRep is the number of repetation for the solver. Takes the best result
    #  @param maxIt is the maximum number of iterations allowed for optimizing community assignments
    def solveCommunityModel(self, K, numRep=1, maxIt=5000):
        self.K = K
        bestZ = 0.
        bestLogL = -1e+10

        #Make repeatation for better inference
        for rep in range(numRep):
            #Decide starting point
            Z = self.initializeCommunities(random=False)
            #Optimize community assignments
            logL0, Z = self.maximizeCommunities(Z, maxIt)

            if logL0 > bestLogL:
                bestZ = Z
                bestLogL = logL0

            activeK = len(np.unique(np.argmax(Z, axis=1)))
            print("K: %d Best log-likelihood: %0.2f"%(activeK, bestLogL))
            print('')

        return bestZ, bestLogL






