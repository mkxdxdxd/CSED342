import shell
import util
import wordsegUtil



############################################################
# Problem 1: Word Segmentation

# Problem 1a: Solve the word segmentation problem under a unigram model

class WordSegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return self.query
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return len(state)==0
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
        #print(state)
        result=[]
        if self.isEnd(state) != 1:
            for i in range(len(state),0,-1):
                action = state[:i]
                remaining = state[len(action):]
                result.append((action, remaining, self.unigramCost(action)))
        #print(result)
        #print("\n\n")
        return result
        # END_YOUR_ANSWER

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch()
    ucs.solve(WordSegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER
    
# Problem 1b: Solve the k-word segmentation problem under a unigram model

class KWordSegmentationProblem(util.SearchProblem):
    def __init__(self, k, query, unigramCost):
        self.k = k
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return self.query
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return len(state)==0
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 7 lines of code, but don't worry if you deviate from this)
        raise NotImplementedError  # remove this line before writing code
        # END_YOUR_ANSWER

def segmentKWords(k, query, unigramCost):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    raise NotImplementedError  # remove this line before writing code
    # END_YOUR_ANSWER

############################################################
# Problem 2: Vowel Insertion

# Problem 2a: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return (self.queryWords[0],0)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[1] == len(self.queryWords) - 1
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 9 lines of code, but don't worry if you deviate from this)
        #print(state)
        result = []
        index = state[1]+1 #index of the word that i want to search for
        possible_fills = self.possibleFills(self.queryWords[index])
        
        if len(possible_fills) == 0: #if possible_fills is 0, update on the index (pass)
            possible_fills.add(self.queryWords[index]) 
        
        for action in possible_fills:
            result.append((action, (action,index), self.bigramCost(state[0],action))) #append every vowel insertion possibilities
            
        return result
        # END_YOUR_ANSWER

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    if len(queryWords)==0:
        return ''
    else:
        queryWords.insert(0,wordsegUtil.SENTENCE_BEGIN)
    ucs=util.UniformCostSearch(verbose=1)
    ucs.solve(VowelInsertionProblem(queryWords,bigramCost,possibleFills))
    words = ' '.join(ucs.actions)
    return words
    # END_YOUR_ANSWER

# Problem 2b: Solve the limited vowel insertion problem under a bigram cost

class LimitedVowelInsertionProblem(util.SearchProblem):
    def __init__(self, impossibleVowels, queryWords, bigramCost, possibleFills):
        self.impossibleVowels = impossibleVowels
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return (self.queryWords[0],0)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[1] == len(self.queryWords) - 1
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 10 lines of code, but don't worry if you deviate from this)
        result = []
        index = state[1]+1 #index of the word that i want to search for
        possible_fills = self.possibleFills(self.queryWords[index])
        possible_fills_limit = set([])
        
        for word in possible_fills:
            if self.impossibleVowels not in word:
                possible_fills_limit.add(word)
        #print("possible_fillers: ",possible_fills_limit)
        
        if len(possible_fills_limit) == 0: #if possible_fills is 0, update on the index (pass)
            possible_fills_limit.add(self.queryWords[index]) 
        
        for action in possible_fills_limit:
            result.append((action, (action,index), self.bigramCost(state[0],action))) #append every vowel insertion possibilities
            
        return result
        # END_YOUR_ANSWER

def insertLimitedVowels(impossibleVowels, queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    if len(queryWords)==0:
        return ''
    else:
        queryWords.insert(0,wordsegUtil.SENTENCE_BEGIN)
        
    ucs=util.UniformCostSearch(verbose=1)
    ucs.solve(VowelInsertionProblem(impossibleVowels,queryWords,bigramCost,possibleFills))
    words = ' '.join(ucs.actions)
    return words
    # END_YOUR_ANSWER

############################################################
# Problem 3: Putting It Together

# Problem 3a: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return (self.query,wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return len(state[0]) == 0
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        result=[]
        #print("state: ", state)
        index = len(state[0])+1 
        current_word = state[0]
        
        for i in range(1,index):
            action=current_word[:i]  
            remain=current_word[i:]
            possible_fills=self.possibleFills(action)
            #print("possible filler: ", possible_fills)
            for word in possible_fills:
                result.append((word, (remain,word), self.bigramCost(state[1],word)))
        #print("result: ", result)
        #print("\n\n")
        return result
        # END_YOUR_ANSWER

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=1)
    ucs.solve(JointSegmentationInsertionProblem(query,bigramCost,possibleFills))
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

############################################################
# Problem 4: A* search

# Problem 4a: Define an admissible but not consistent heuristic function

class SimpleProblem(util.SearchProblem):
    def __init__(self):
        # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
         self.start = 'A'
        # END_YOUR_ANSWER

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return self.start
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == 'E'
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
        if state == 'A':
            return [('A->B','B',1),('A->C','C',1)]
        elif state == 'B':
            return [('B->D','D',1)]
        elif state == 'C':
            return [('C->D','D',2)]
        elif state == 'D':
            return [('D->E','E',1000)]
        # END_YOUR_ANSWER

def admissibleButInconsistentHeuristic(state):
    # BEGIN_YOUR_ANSWER (our solution is 2 lines of code, but don't worry if you deviate from this)
    h = {'A':0, 'B':1001 , 'C':0, 'D':0, 'E':0}
    return h[state]
    # END_YOUR_ANSWER

# Problem 4b: Apply a heuristic function to the joint segmentation-and-insertion problem

def makeWordCost(bigramCost, wordPairs):
    """
    :param bigramCost: learned bigram cost from a training corpus
    :param wordPairs: all word pairs in the training corpus
    :returns: wordCost, which is a function from word to cost
    """
    # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
    cost = {}
    for w, w_ in wordPairs:
        cost[w] = -1
    for w, w_ in wordPairs:
        cost[w] = min(bigramCost(w,w_),cost[w])
    
    def wordCost(w):
        if w not in cost:
            return bigramCost(wordsegUtil.SENTENCE_UNK,w)
        else:
            return cost[w]
    # END_YOUR_ANSWER

class RelaxedProblem(util.SearchProblem):
    def __init__(self, query, wordCost, possibleFills):
        self.query = query
        self.wordCost = wordCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return self.query
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[1] == len(self.query)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
        result = []
        pair = {}
        #print("state: ", state)
        index = len(state[0])+1 
        current_word = state[0]
        
        for i in range(1,index):
            action=current_word[:i]  
            remain=current_word[i:]
            possible_fills=self.possibleFills(action)
    
            for word in possible_fills:
                pair['word'] = self.wordCost(word)
            for word in possible_fills:
                next_word, cost = min(pair.items(), key = lambda x:x[1])
                result.append((next_word, (remain,), cost))

        return result
        # END_YOUR_ANSWER

def makeHeuristic(query, wordCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 2 lines of code, but don't worry if you deviate from this)
    dp = util.DynamicProgramming(RelaxedProblem(query, wordCost, possibleFills))
    ret = {}
    def heuristic(state):
        if state[0] not in ret:
            ret[state[0]] = dp(state) 
        return ret[state[0]]
    # END_YOUR_ANSWER

def fastSegmentAndInsert(query, bigramCost, wordCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
    heuristic = makeHeuristic(query, wordCost, possibleFills)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills), heuristic)
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

############################################################

if __name__ == '__main__':
    shell.main()
