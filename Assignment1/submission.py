import collections
import math

############################################################
# Problem 1a
def denseVectorDotProduct(v1, v2):    
    """
    Given two dense vectors |v1| and |v2|, each represented as list,
    return their dot product.
    You might find it useful to use sum(), and zip() and a list comprehension.
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    ans = 0
    n = len(v1)
    for i in range(n):
        ans += v1[i]*v2[i]
    return ans
    # END_YOUR_ANSWER

############################################################
# Problem 1b
def incrementDenseVector(v1, scale, v2):
    """
    Given two dense vectors |v1| and |v2| and float scalar value scale, return v = v1 + scale * v2.
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    n = len(v1)
    ans = [0] * n
    for i in range(n):
        ans[i] += v1[i]+ scale*v2[i]
    return ans
    # END_YOUR_ANSWER

############################################################
# Problem 1c
def dense2sparseVector(v):
    """
    Given a dense vector |v|, return its sparse vector form,
    represented as collection.defaultdict(float).
    
    For exapmle:
    >>> dv = [0, 0, 1, 0, 3]
    >>> dense2sparseVector(dv)
    # defaultdict(<class 'float'>, {2: 1, 4: 3})
    
    You might find it useful to use enumerate().
    """
    # BEGIN_YOUR_ANSWER 
    ans = {}
    for index in range(len(v)):
        if v[index] == 0: continue
        else: ans[index] = v[index]
    return ans
    # END_YOUR_ANSWER
    
############################################################
# Problem 1d
def sparseVectorDotProduct(v1, v2):  # -> sparse vector product, dense vectoer product, dense sparse matmul
    """
    Given two sparse vectors |v1| and |v2|, each represented as collection.defaultdict(float),
    return their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    # note that v1 and v2 are in a dictionary format {key: value}
    # search for the common key value with for and if sentence and sum altogether
    return sum([(v1[key]*v2[key]) for key in v1 if key in v2])
    # END_YOUR_ANSWER

############################################################
# Problem 1e
def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, return v = v1 + scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
    for key in v2:
        v1[key]+=scale*v2[key]
    return v1
    # END_YOUR_ANSWER

############################################################
# Problem 2a
def minkowskiDistance(loc1, loc2, p = math.inf): 
    """
    Return the Minkowski distance for p between two locations,
    where the locations are n-dimensional tuples.
    the Minkowski distance is generalization of
    the Euclidean distance and the Manhattan distance. 
    In the limiting case of p -> infinity,
    the Chebyshev distance is obtained.
    
    For exapmle:
    >>> p = 1 # manhattan distance case
    >>> loc1 = (2, 4, 5)
    >>> loc2 = (-1, 3, 6)
    >>> minkowskiDistance(loc1, loc2, p)
    # 5

    >>> p = 2 # euclidean distance case
    >>> loc1 = (4, 4, 11)
    >>> loc2 = (1, -2, 5)
    >>> minkowskiDistance = (loc1, loc2)  # 9

    >>> p = math.inf # chebyshev distance case
    >>> loc1 = (1, 2, 3, 1)
    >>> loc2 = (10, -12, 12, 2)
    >>> minkowskiDistance = (loc1, loc2, math.inf)
    # 14
    
    """
    # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
    if p == math.inf:#special case where p == infinity
        _abs = 0
        for a, b in zip(loc1, loc2):
            _abs = max(_abs, abs(a-b))
        ans = _abs
    else:# for normal cases where p is an element of the set of the real numbers
        _power = 0
        _sum = 0
        for a, b in zip(loc1, loc2):
            _power = pow(abs(a-b), p)
            _sum += _power
        ans = pow(_sum, 1/p)
        
    return ans

    # END_YOUR_ANSWER

############################################################
# Problem 2b
def getLongestWord(text):
    """
    Given a string |text|, return the longest word in |text|. 
    If there are ties, choose the word that comes first in the alphabet.
    
    For example:
    >>> text = "tiger cat dog horse panda"
    >>> getLongestWord(text) # 'horse'
    
    Note:
    - Assume there is no punctuation and no capital letters.
    
    Hint:
    - max/min function returns the maximum/minimum item with respect to the key argument.
    """

    # BEGIN_YOUR_ANSWER (our solution is 4 line of code, but don't worry if you deviate from this)
    dic = {}
    x = text.split()
    for i in range(len(x)):
        dic[x[i]] = len(x[i])
    
    #find max length of a word
    _max = -1
    for key in dic:
        if dic[key]>_max: _max = dic[key]
    
    #find the answer, comparing the alphabet
    #initialise
    for key in dic:
        if dic[key] == _max:
            ans = key
            break
    #compare
    for key in dic:
        if dic[key] == _max and key<ans: 
            ans = key
            
    return ans
    # END_YOUR_ANSWER

############################################################
# Problem 2c
def getFrequentWords(text, freq):
    """
    Splits the string |text| by whitespace
    and returns a set of words that appear at a given frequency |freq|.
    """
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    dic = {}
    x = text.split()
    #initialise
    for i in range(len(x)):
        dic[x[i]] =  0
        
    #counting
    for i in range(len(x)):
        dic[x[i]] =  dic[x[i]]+1
    #find words with a given freq
    ans = set([])
    for key in dic:
        if dic[key] == freq:
            ans.update([key])
    return ans
    # END_YOUR_ANSWER 