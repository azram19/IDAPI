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
    
    prior_sum = sum(prior)
    # Normalize
    if prior_sum != 0:
        prior /= prior_sum
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
            if nP[j] != 0:
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
        norm_sum = sum(aJPT[:, i])
        if norm_sum != 0:
            aJPT[:, i] /= norm_sum
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
    norm_sum = sum(rootPdf)
    
    if norm_sum != 0:
        rootPdf /= norm_sum
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
        for j in xrange(nsp1):
            for k in xrange(nsp2):
                for l in xrange(theData.shape[0]):
                    cPT[i, j, k] += (theData[l, child] == i and
                                theData[l, parent1] == j and
                                theData[l, parent2] == k)

    #N(p1&p2)
    for j in xrange(nsp1):
        for k in xrange(nsp2):
            for l in xrange(theData.shape[0]):
                nP[j, k] += (
                            theData[l, parent1] == j and
                            theData[l, parent2] == k)

    

    #P(c|p1&p2) = N(c1&p1&p2) / N(p1&p2)
    for i in xrange(nsc):
        for j in xrange(nsp1):
            for k in xrange(nsp2):
                if nP[j, k] != 0:
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
def HepatitisCBayesianNetwork(theData, noStates):
    arcList = [[0],[1],[2,0],[3,4],[4,1],[5,4],[6,1],[7,0,1],[8,7]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT(theData, 3, 4, noStates)
    cpt4 = CPT(theData, 4, 1, noStates)
    cpt5 = CPT(theData, 5, 4, noStates)
    cpt6 = CPT(theData, 6, 1, noStates)
    cpt7 = CPT_2(theData, 7, 0, 1, noStates)
    cpt8 = CPT(theData, 8, 7, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5, cpt6, cpt7, cpt8]
    return arcList, cptList
# end of coursework 3 task 2
#
# Function to calculate the MDL size of a Bayesian Network
def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
    Bn = 0.0
# Coursework 3 task 3 begins here
    # arcList connectivity - [0] is node [1:] are parents
    # cptList, conditional probability table for each nodes
    # N is the number of cases used to compute the probabilites
    N = noDataPoints

    # Compute number of parameters required to get probabilities for a node
    for i in xrange(len(arcList)):
        n = arcList[i][0]
        sh = cptList[n].shape

        shp = list(sh)
        shp[0] -= 1
        Bn += prod(shp)

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
    # Get conditional probability for a node
    def getProbability(node):
        # Get correct probability table
        cpt = cptList[node]

        # Create index to the right entry in the table
        cptIndices = []
        for n in arcList[node]:
            cptIndices.append(dataPoint[n])

        # Return probablity
        prob = cpt[tuple(cptIndices)]
        return prob

    nodesToTraverse = range(len(arcList))
    while nodesToTraverse:
        # Get the first node in the queue 
        node = nodesToTraverse.pop(0)
        # Get probability
        jP *= getProbability(node)

# Coursework 3 task 4 ends here 
    return jP
#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy=0
# Coursework 3 task 5 begins here
    for i in xrange(theData.shape[0]):
        jP = JointProbability(theData[i], arcList, cptList)
        if(jP != 0):
            mdlAccuracy += log2(jP)

# Coursework 3 task 5 ends here 
    return mdlAccuracy
    
#Coursework 3 task 6 begins here    
def BestScoringNet(theData, arcList, cptList, noDataPoints, noStates):
    import copy
    
    def buildNetwork(arcList):
        cptList = []
        for nodes in arcList:
            node = nodes[0]
            if(len(nodes) == 1):
                cpt = Prior(theData, node, noStates)
            elif(len(nodes) == 2):
                cpt = CPT(theData, node, nodes[1], noStates)
            else:
                cpt = CPT_2(theData, node, nodes[1], nodes[2], noStates)

            cptList.append(cpt)
        return arcList, cptList

    nnodes = len(arcList)
    init = True
    bestMDLScore = 0
    bestArcList = []
    bestCptList = []
    # Iterate over all possible arcs
    for arc in map(tuple, ([n1, n2] for n1 in range(nnodes) for n2 in range(nnodes) if n1 != n2)):
        # Check if this arc exists
        if arc[1] in arcList[arc[0]]:
            # Make a copy of the list
            newArcList = copy.deepcopy(arcList)
            # Remove the arc
            newArcList[arc[0]].remove(arc[1])

            # Build network
            _, newCptList = buildNetwork(newArcList)

            # Compute mdlScore for the new network
            mdlScore = MDLSize(newArcList, newCptList, noDataPoints, noStates) - MDLAccuracy(theData, newArcList, newCptList) 

            # Save the best results (if there is a previous result)
            if mdlScore < bestMDLScore or init:
                init = False
                bestMDLScore = mdlScore
                bestArcList = newArcList
                bestCptList = newCptList

    return bestArcList, bestCptList

#
# End of coursework 3
#
# Coursework 4 begins here
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
# main program part for Coursework 3
#
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
AppendString("results3.txt","Coursework Three Results by Lukasz Koprowski (lk1510) & Agata Mosinska (am9310)")
AppendString("results3.txt","\nMDLSize of network")
(arcList, cptList) = HepatitisCBayesianNetwork(theData, noStates)
size = MDLSize(arcList, cptList, noDataPoints, noStates)
AppendString("results3.txt", size)
AppendString("results3.txt", "\nMDLAccuracy")
accuracy = MDLAccuracy(theData, arcList, cptList)
AppendString("results3.txt", accuracy)
AppendString("results3.txt", "\nMDLScore")
AppendString("results3.txt", - accuracy + size)
AppendString("results3.txt", "\nBest Scoring Network")
(newArcs,newCpts) = BestScoringNet(theData,arcList,cptList,noDataPoints, noStates)
bestScoreRemoved = - MDLAccuracy(theData, newArcs, newCpts) + MDLSize(newArcs, newCpts, noDataPoints, noStates)
AppendString("results3.txt", "Score of the best network")
AppendString("results3.txt", bestScoreRemoved)
