#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
import math
import operator
#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float )
# Coursework 1 task 1 should be inserted here
    
    for i in xrange(theData.shape[0]):
        dp = theData[i, :] # Select a datapoint
        s = dp[root]    # State of the variable
        prior[s] += 1   # Increment

    # Normalize
    prior /= sum(prior)
# end of Coursework 1 task 1
    return prior
# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float)
# Coursework 1 task 2 should be inserte4d here
    nP = zeros((noStates[varP]), float)
    # N(c2&d4)
    for i in xrange(noStates[varC]):
        for j in xrange(noStates[varP]):
            for k in xrange(theData.shape[0]):
                cPT[i, j] += (theData[k, varC] == i and theData[k, varP] == j)
    # N(c2)
    for j in xrange(noStates[varP]):
        for k in xrange(theData.shape[0]):
            nP[j] += (theData[k, varP] == j)

    # P(d4|c2) = N(c2&d4) / N(c2)
    for i in xrange(noStates[varC]):
        for j in xrange(noStates[varP]):
            cPT[i, j] /= nP[j] 
# end of coursework 1 task 2
    return cPT
# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
#Coursework 1 task 3 should be inserted here 
    # P(c2&d4) == N(c2&d4) / len(theData)
    # N(c2&d4)
    for i in xrange(noStates[varRow]):
        for j in xrange(noStates[varCol]):
            for k in xrange(theData.shape[0]):
                jPT[i, j] += (theData[k, varRow] == i and theData[k, varCol] == j)
            # Normalize by the number of data points
            jPT[i, j] /= theData.shape[0] 
     
# end of coursework 1 task 3
    return jPT
#
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
#Coursework 1 task 4 should be inserted here 
    # Normalize columns to make them sum to 1
    for i in xrange(aJPT.shape[1]):  
        aJPT[:, i] /= sum(aJPT[:, i])
# coursework 1 taks 4 ends here
    return aJPT

#
# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes): 
    rootPdf = zeros((naiveBayes[0].shape[0]), float)
# Coursework 1 task 5 should be inserted here
    # naive Bayes - [Prior, cpt1, cpt2, cpt3]
    # the Query - instantiated states [....]
    for i in xrange(naiveBayes[0].shape[0]):
        # Prior for ci 
        rootPdf[i] = naiveBayes[0][i]
        for j in xrange(1, len(theQuery)+1):
            # CPT - j | ci
            rootPdf[i] *= naiveBayes[j][theQuery[j - 1],i]

    # Normalize
    rootPdf /= sum(rootPdf)

# end of coursework 1 task 5
    return rootPdf
#
# End of Coursework 1
#
# Coursework 2 begins here
#
# Calculate the mutual information from the joint probability table of two variables
def MutualInformation(jP):
    mi=0.0
# Coursework 2 task 1 should be inserted here
   
    for i in xrange(jP.shape[0]):
        for j in xrange(jP.shape[1]):
            if(jP[i,j]!=0):
                mi += jP[i,j] * math.log((jP[i,j] /(sum(jP[i,:])*sum(jP[:,j]))))/math.log(2)

# end of coursework 2 task 1
    return mi
#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
# Coursework 2 task 2 should be inserted here
    for i in xrange(noVariables):
        for j in xrange(noVariables):
            jPT = JPT(theData,i,j,noStates)
            mutInfo = MutualInformation(jPT)
            MIMatrix[i,j] = MIMatrix[j,i] = mutInfo
            if (i==j):
                MIMatrix[i,j] = MIMatrix[j,i] = 0

# end of coursework 2 task 2
    return MIMatrix
# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here
    copMatrix = depMatrix
    
    while (copMatrix.sum()!=0):
        #var1, var2 = max(enumerate(copMatrix), key=operator.itemgetter(1))
        maxNum = 0
        for i in xrange(copMatrix.shape[0]):
            for j in xrange(copMatrix.shape[1]):
                if(copMatrix[i,j]>maxNum):
                    maxNum = copMatrix[i,j]
                    node1 = i
                    node2 = j
        
        entry = [maxNum,node1,node2]
        depList = depList+[entry]
        copMatrix[node1,node2] = copMatrix[node2, node1] = 0
        
# end of coursework 2 task 3
    return array(depList)
#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4

def Root(unions, b):
    """Return the root set of an item
    Re-parents nodes while traversing them to minimize depth of a tree"""

    root = unions[b]

    while root != unions[root]:
        oldRoot = root
        root = unions[root]
        unions[oldRoot] = root

    unions[b] = root
    return root

def Unify(unions, a, b):
    """Merge two items into the same set"""
    rootA = Root(unions, a)
    rootB = Root(unions, b)

    unions[rootA] = rootB

def SpanningTreeAlgorithm(depList, noVariables):
    # unions[i] - root of a set tree to which the node belongs
    unions = zeros(noVariables, dtype=float32)

    spanningTree = []
    # Assume it is sorted
    sortedDepList = depList
    # Set each variable to be in its own set
    for i in xrange(noVariables):
        unions[i] = i

    # Iterate over the list of `dependencies`
    for i in xrange(sortedDepList.shape[0]):
        dep = sortedDepList[i, :]
        rootA = Root(unions, dep[1])
        rootB = Root(unions, dep[2])

        # To avoid loops check if variables do not belong to the same set
        if(rootA != rootB):
            Unify(unions, dep[1], dep[2])
            spanningTree.append(dep)

    return array(spanningTree)
#
# End of coursework 2
#
# Coursework 3 begins here
#
# Function to compute a CPT with multiple parents from he data set
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT_2(theData, child, parent1, parent2, noStates):
    cPT = zeros([noStates[child],noStates[parent1],noStates[parent2]], float )
# Coursework 3 task 1 should be inserted here
    # Number of states for {}
    nsc = noStates[child]
    nsp1 = noStates[parent1]
    nsp2 = noStates[parent2]
    nP = zeros([noStates[parent1], noStates[parent2]], float)
    #N(c&p1&p2)
    for i in xrange(nsc):
        for j in xrange{nsp1}:
            for k in xrange(nsp2):
                for l in xrange(theData.shape[0]):
                    cPT[i, j, k] += (theData[l, child] == i and
                                theData[l, parent1] == j and
                                theData[l, parent2] == k)

    #N(p1&p2)
    for j in xrange{nsp1}:
        for k in xrange(nsp2):
            for l in xrange(theData.shape[0]):
                nP[j, k] += (
                            theData[l, parent1] == j and
                            theData[l, parent2] == k)

    

    #P(c|p1&p2) = N(c1&p1&p2) / N(p1&p2)
    for i in xrange(nsc):
        for j in xrange{nsp1}:
            for k in xrange(nsp2):
                cPT[i, j, k] /= nP[j, k]
# End of Coursework 3 task 1           
    return cPT
#
# Definition of a Bayesian Network
def ExampleBayesianNetwork(theData, noStates):
    arcList = [[0],[1],[2,0],[3,2,1],[4,3],[5,3]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT_2(theData, 3, 2, 1, noStates)
    cpt4 = CPT(theData, 4, 3, noStates)
    cpt5 = CPT(theData, 5, 3, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5]
    return arcList, cptList
# Coursework 3 task 2 begins here

# end of coursework 3 task 2
#
# Function to calculate the MDL size of a Bayesian Network
def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
# Coursework 3 task 3 begins here
    # arcList connectivity - [0] is node [1:] are parents
    # cptList, conditional probability table for each nodes
    # N is the number of cases used to compute the probabilites
    N = noDataPoints

    # Compute number of parameters required to get probabilities for a node
    for i in xrange(len(arcList)):
        n = arcList[i][0]

        # Conditional probabilities
        # Number of entries 
        sh = cptList[n].shape
        # We do not need info from last row
        sh[0] -= 1

        Bn += prod(sh)

    mdlSize = abs(Bn)
    mdlSize *= log2(N)
    mdlSize /= 2
    
# Coursework 3 task 3 ends here 
    return mdlSize 
#
# Function to calculate the joint probability of a single data point in a Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here
    
    def getVProbability(i, s):
        p = 1
        p *= sum(cptList[i][s])
        return p

    for i, s in enumerate(dataPoint):
        jP *= getVProbability(i, s)

# Coursework 3 task 4 ends here 
    return jP
#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy=0
# Coursework 3 task 5 begins here
    
    
    mdlAccuracy = log2(mdlAccuracy)
# Coursework 3 task 5 ends here 
    return mdlAccuracy
#
# End of coursework 2
#
# Coursework 3 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    # Coursework 4 task 1 begins here



    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here


    # Coursework 4 task 2 ends here
    return covar
def CreateEigenfaceFiles(theBasis):
    adummystatement = 0 #delete this when you do the coursework
    # Coursework 4 task 3 begins here

    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here

    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags):
    adummystatement = 0  #delete this when you do the coursework
    # Coursework 4 task 5 begins here

    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes

    
    # Coursework 4 task 6 ends here
    return array(orthoPhi)

#
# main program part for Coursework 1
#
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
AppendString("results2.txt","Coursework One Results by Lukasz Koprowski (lk1510)& Agata Mosinska (am9310)")
AppendString("results2.txt","") #blank line
AppendString("results2.txt", "Dependency Matrix")
depMatrix = DependencyMatrix(theData,noVariables,noStates)
AppendString("results2.txt","")
AppendArray("results2.txt",depMatrix)
AppendString("results2.txt","")
AppendString("results2.txt","Dependency List")
depList = DependencyList(depMatrix)
AppendArray("results2.txt",depList)
spanningTree = SpanningTreeAlgorithm(depList, noVariables)
AppendString("results2.txt","") #blank line
AppendString("results2.txt","Maximally Weighted Spanning Tree") #blank line
AppendArray("results2.txt", spanningTree)
