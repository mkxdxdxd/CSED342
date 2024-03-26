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
        return 0
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code  , but don't worry if you deviate from this)
        return len(self.query)==state
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
        results = []
        start = state+1
        end = len(self.query) + 1
        
        for i in range(start, end):
            action = self.query[state:i]
            results.append((action, i, self.unigramCost(action)))
        return results
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
        return (0,0)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[0] == len(self.query)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 7 lines of code, but don't worry if you deviate from this)
        results=[]
        start = state[0] + 1
        end = len(self.query) + 1
        
        for i in range(start, end):
            action = self.query[state[0]:i]
            if state[1] < self.k: #check wheter the sementation is over k
                results.append((action, (i, state[1]+1), self.unigramCost(action)))
        return results
        # END_YOUR_ANSWER

def segmentKWords(k, query, unigramCost):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(KWordSegmentationProblem(k, query, unigramCost))

    return ' '.join(ucs.actions)
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
        return (wordsegUtil.SENTENCE_BEGIN,0)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[1] == len(self.queryWords)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 9 lines of code, but don't worry if you deviate from this)
        result = []
        possible_fills = self.possibleFills(self.queryWords[state[1]])
        print("state", state,"\n")
        if len(possible_fills) == 0: #if possible_fills is 0, update on the index (pass)
            possible_fills.add(self.queryWords[state[1]]) 
        
        for action in possible_fills:
            result.append((action, (action,state[1]+1), self.bigramCost(state[0],action))) #append every vowel insertion possibilities
        return result
        # END_YOUR_ANSWER

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    if len(queryWords)==0:
        return ''
    else:
        queryWords.insert(0,wordsegUtil.SENTENCE_BEGIN)
    ucs=util.UniformCostSearch(verbose=0)
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

        impossibleV = [x for x in self.impossibleVowels] #put all impossible vowel into a list
        
        def vowelcontaining(v, w): #check is a word w contains v
            return v in w
        
        for vowel in impossibleV:   
            possible_fills = [word for word in possible_fills if not vowelcontaining(vowel, word)] #if containing an impossible vowel, put it out from the possible_fills
        
        possible_fills = set(possible_fills) #remove multiplicities

        if len(possible_fills) == 0: #if possible_fills is 0, update on the index (pass)
            possible_fills.add(self.queryWords[index])
        
        for action in possible_fills:
            result.append((action, (action,index), self.bigramCost(state[0],action))) #append every vowel insertion possibilities  
        return result
        # END_YOUR_ANSWER

def insertLimitedVowels(impossibleVowels, queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    if len(queryWords)==0:
        return ''
    else:
        queryWords.insert(0,wordsegUtil.SENTENCE_BEGIN)
    ucs=util.UniformCostSearch(verbose=0)
    ucs.solve(LimitedVowelInsertionProblem(impossibleVowels,queryWords,bigramCost,possibleFills))
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
        return wordsegUtil.SENTENCE_BEGIN,0
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[1] == len(self.query)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        currentWord = state[0]
        index = state[1]
        end = len(self.query) + 1

        result = []
        for i in range(index + 1, end):
            word = self.query[index:i] #segment
            actions = self.possibleFills(word) #filler function for the segment
            for action in actions:
                result.append((action, (action, i), self.bigramCost(currentWord, action)))
        return result
        # END_YOUR_ANSWER

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
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
    # BEGIN_YOUR_ANSWER (our solution is 12 lines of code, but don't worry if you deviate from this)
    costs = {}

    for word1, word2 in wordPairs:
        cost = bigramCost(word1, word2)
        costs[word2] = min(cost ,costs.get(word2, cost)) 

    def wordCost(word):
        return costs.get(word, bigramCost(wordsegUtil.SENTENCE_UNK, word))

    return wordCost
    # END_YOUR_ANSWER

class RelaxedProblem(util.SearchProblem):
    def __init__(self, query, wordCost, possibleFills):
        self.query = query
        self.wordCost = wordCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return None, 0
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[1] == len(self.query)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
        index = state[1]
        results = []
        for i in range(index + 1, len(self.query) + 1):
            word = self.query[index:i]
            actions =  self.possibleFills(word)
            for action in actions:
                results.append((None, (None, i), self.wordCost(action)))
        return results
        # END_YOUR_ANSWER

def makeHeuristic(query, wordCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 7 lines of code, but don't worry if you deviate from this)
    # dp로 heuristic을 pre-define한다. 그리고 나서 heuristic fuction을 사용해서 A* 알고리즘을 수행한다.
    dp = util.DynamicProgramming(RelaxedProblem(query, wordCost, possibleFills))
    value = []
    for i in range(len(query)+1):
        value.append(dp((None, i)))
    def heuristic(state):
        return value[state[1]]

    return heuristic
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


