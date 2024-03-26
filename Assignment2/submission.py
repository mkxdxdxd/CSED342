#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *


############################################################
# Problem 1: hinge loss
############################################################

def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        so, touching, quite, impressive, not, boring
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'so': 1,'touching': 1, 'quite': 0, 'impressive': 0, 'not': -1,'boring':-1}
    # END_YOUR_ANSWER

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    dic = {}
    x = x.split()
    #initialise
    for i in range(len(x)):
        dic[x[i]] =  0
    #counting
    for i in range(len(x)):
        dic[x[i]] =  dic[x[i]]+1
    return dic
    # END_YOUR_ANSWER

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    '''
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)

    #calculate
    deri = {}
    for x,y in trainExamples:
        for i in featureExtractor(x):
            weights[i]=0
            deri[i]=0
            
    def nll_deri(phi,sig_wphi,y):
        if (y == 1):
            for i in phi:
                deri[i] = (sig_wphi - 1)*phi[i]
        elif (y == -1):
            for i in phi:
                deri[i] = sig_wphi*phi[i]
        return deri
        
    for i in range(numIters):
        for x, y in trainExamples:
            phi = featureExtractor(x)
            sig_wphi = sigmoid(dotProduct(weights, phi))
            for j in featureExtractor(x):
                    weights[j] = weights[j] - eta*nll_deri(phi,sig_wphi,y)[j]
    
    # END_YOUR_ANSWER
    return weights

############################################################
# Problem 2c: ngram features

def extractNgramFeatures(x, n):
    """
    Extract n-gram features for a string x
    
    @param string x, int n: 
    @return dict: feature vector representation of x. (key: n consecutive word (string) / value: occurrence)
    
    For example:
    >>> extractNgramFeatures("I am what I am", 2)
    {'I am': 2, 'am what': 1, 'what I': 1}

    Note:
    There should be a space between words and NO spaces at the beginning and end of the key
    -> "I am" (O) " I am" (X) "I am " (X) "Iam" (X)

    Another example
    >>> extractNgramFeatures("I am what I am what I am", 3)
    {'I am what': 2, 'am what I': 2, 'what I am': 2}
    """
    # BEGIN_YOUR_ANSWER (our solution is 12 lines of code, but don't worry if you deviate from this)
    words = x.split()
    split_n = []
    temp = []
    phi = {}
    
    for i in range(len(words) - n+1):
        split_n.append(words[i:i+n])
        
    for i in range(len(words) - n+1):
        temp.append(" ".join(split_n[i]))     
    #search
    phi = dict(Counter(temp))
    # END_YOUR_ANSWER
    return phi

############################################################
# Problem 3a: k-means exercise
############################################################

def problem_3a_1():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu_x' and 'mu_y'.
    Assume the initial centers are
    ({'mu_x': -2, 'mu_y': 0}, {'mu_x': 3, 'mu_y': 0})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return ({'mu_x': -0.5, 'mu_y': 1.5}, {'mu_x': 3, 'mu_y': 1.5})
    # END_YOUR_ANSWER

def problem_3a_2():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu_x' and 'mu_y'.
    Assume the initial centers are
    ({'mu_x': -1, 'mu_y': -1}, {'mu_x': 2, 'mu_y': 3})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return ({'mu_x': -1, 'mu_y': 0}, {'mu_x': 2, 'mu_y': 2})
    # END_YOUR_ANSWER

############################################################
# Problem 3: k-means implementation
############################################################

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_ANSWER (our solution is 40 lines of code, but don't worry if you deviate from this)
    centroid_dot = []
    centroids = random.sample(examples, K) #randomly choose K data from the 'examples' and make it as centroids
    for c in centroids:
        dot = dotProduct(c,c)
        centroid_dot.append(dot) #centroid^2 in a list
    
    # calculate the distance between the sample and centroid
    example_dot = []
    for e in examples:
        dot = dotProduct(e, e)
        example_dot.append(dot) #example^2 in a list
        
    def distance(ic, ie):  #x^2+y^2
        return centroid_dot[ic] + example_dot[ie] - 2 * dotProduct(centroids[ic], examples[ie])
    
    #start the k-means algorithm
    assignments=[]
    distances = {}

    for _ in range(maxIters):
        group_id = []
        #1. calculate which group does the samples belong to
        for i in range(len(examples)):
            #calculate distance between the sample and the centroids
            for j in range(K):
                 distances[j] = distance(j,i)
            minimum = min(distances, key = distances.get)
            group_id.append(minimum)
            
        #terminating when the group_id and assignment converge   
        if group_id == assignments:
            break
        assignments = group_id
        
        #2. update centroids
        #sum all elements by group
        group_sum = [[{}, 0] for _ in range(K)] 
        #centroid1 [{0: sum, 1: sum ...},count]
        #centroid2 [{0: sum, 1: sum ...},count]
        
        #find sum for each x1 =.. x2 = ...
        for i in range(len(examples)):
            increment(group_sum[assignments[i]][0], 1, examples[i])
            group_sum[assignments[i]][1] += 1
        
        #update on the average pt
        for i, (center, n) in enumerate(group_sum):
            if n > 0:
                for h, g in list(center.items()):
                    center[h] = g / n
            centroids[i] = center #update on the new center
            centroid_dot[i] = dotProduct(center, center) # calcualte the new centoid for the next iteration
            
    #calculate loss
    loss = 0
    for i in range(len(examples)):
        loss += distance(assignments[i], i)
    return (centroids, assignments, loss)
    # END_YOUR_ANSWER

