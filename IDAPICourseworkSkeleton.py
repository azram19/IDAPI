#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
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
   

# end of coursework 2 task 1
    return mi
#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
# Coursework 2 task 2 should be inserted here
    

# end of coursework 2 task 2
    return MIMatrix
# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here
    

# end of coursework 2 task 3
    return array(depList2)
#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4

def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
  
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


# Coursework 3 task 3 ends here 
    return mdlSize 
#
# Function to calculate the joint probability of a single data point in a Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here


# Coursework 3 task 4 ends here 
    return jP
#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy=0
# Coursework 3 task 5 begins here


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
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("Neurones.txt")
theData = array(datain)
AppendString("results.txt","Coursework One Results by Lukasz Koprowski & Agata Mosinska")
AppendString("results.txt","") #blank line
AppendString("results.txt","The prior probability of node 0")
prior = Prior(theData, 0,noStates)
AppendList("results.txt",prior)
AppendString("results.txt","") #blank line
AppendString("results.txt","The conditional probability P(2|0)")
condProb = CPT(theData,2,0,noStates)
AppendString("results.txt",condProb)
AppendString("results.txt","") #blank line
AppendString("results.txt","The joint probability P(2&0)")
jointProb = JPT(theData,2,0,noStates)
AppendString("results.txt",jointProb)
AppendString("results.txt","") #blank line
AppendString("results.txt","The conditional probability P(2|0) from joint probability")
jointProb2 = JPT2CPT(jointProb)
AppendString("results.txt",jointProb2)
AppendString("results.txt","")
AppendString("results.txt","Probability of Query 1")

naiveBayes = []
naiveBayes.append(prior)
for i in xrange(1,noVariables):
    naiveBayes.append(CPT(theData,i,0,noStates))

#naiveBayes = [prior,CPT(theData,1,0,noStates),CPT(theData,2,0,noStates),CPT(theData,3,0,noStates),CPT(theData,4,0,noStates),CPT(theData,5,0,noStates) ]
query1 = Query([4,0,0,0,5],naiveBayes)


AppendString("results.txt",query1)
AppendString("results.txt","")
AppendString("results.txt","Probability of Query 2")

query2 = Query([6,5,2,5,5],naiveBayes)
AppendString("results.txt",query2)


