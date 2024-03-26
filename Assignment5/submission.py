from util import manhattanDistance
from game import Directions
import random, util

from game import Agent



class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """
  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex): agent가 수행 가능한 모든 action을 return한다.
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP: 
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action): 
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    # BEGIN_YOUR_ANSWER
    def Vminimax(gameState, agent, depth):
      actions = gameState.getLegalActions(agent)
      if gameState.isWin() or gameState.isLose() or actions == []:
        return gameState.getScore(), None
      
      if depth == 0:
        return self.evaluationFunction(gameState), None
      
      if agent < gameState.getNumAgents() - 1: #next is ghost
        nextAgent = agent + 1
        nextDepth = depth
      else: #next is pacman
        nextAgent = 0 
        nextDepth = depth - 1
        
      Q_value = [(Vminimax(gameState.generateSuccessor(agent, a), nextAgent, nextDepth)[0], a) 
                 for a in actions]
      print(Q_value)
      if agent == self.index: return max(Q_value)
      else: return min(Q_value)
    #print("initialstate", Vminimax(gameState, self.index, self.depth)[0])
    return Vminimax(gameState, self.index, self.depth)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER
    succ = gameState.generateSuccessor(0, action)

    def V(succ, agent, depth):
      
      actions = succ.getLegalActions(agent)
      
      if succ.isWin() or succ.isLose() or actions == []:
        return succ.getScore()
      
      if depth == 0:
        return self.evaluationFunction(succ)
      
      if agent < succ.getNumAgents() - 1: #next is ghost
        nextAgent = agent + 1
        nextDepth = depth
      else: #next is pacman
        nextAgent = 0 
        nextDepth = depth - 1
        
      Qvalue = [V(succ.generateSuccessor(agent, action), nextAgent, nextDepth) for action in actions]
      
      if agent == self.index: return max(Qvalue)
      else: return min(Qvalue)
    
    return V(succ, self.index + 1, self.depth)
    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def Vexpectimax(gameState, agent, depth):
      actions = gameState.getLegalActions(agent)
      if gameState.isWin() or gameState.isLose() or actions == []:
        return gameState.getScore(), None
      
      if depth == 0:
        return self.evaluationFunction(gameState), None
      
      if agent < gameState.getNumAgents() - 1: #next turn is ghost
        nextAgent = agent + 1
        nextDepth = depth
      else: # next turn is pacman
        nextAgent = 0
        nextDepth = depth - 1

      
      if agent == self.index: 
        return max((Vexpectimax(gameState.generateSuccessor(agent, a), nextAgent, nextDepth)[0],a) for a in actions)
      else: 
        return sum(Vexpectimax(gameState.generateSuccessor(agent, a), nextAgent, nextDepth)[0]/len(actions) for a in actions), None
      
    return Vexpectimax(gameState, self.index, self.depth)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    succ = gameState.generateSuccessor(0, action)
    def V(succ, agent, depth):
      actions = succ.getLegalActions(agent)
      if succ.isWin() or succ.isLose() or actions == []:
        return succ.getScore()
      
      if depth == 0:
        return self.evaluationFunction(succ)
      
      if agent < succ.getNumAgents() - 1: #next turn is ghost
        nextAgent = agent + 1
        nextDepth = depth
      else: # next turn is pacman
        nextAgent = 0
        nextDepth = depth - 1

      
      if agent == self.index: 
        return max(V(succ.generateSuccessor(agent, a), nextAgent, nextDepth) for a in actions)
      else: 
        return sum(V(succ.generateSuccessor(agent, a), nextAgent, nextDepth)/len(actions) for a in actions)
      
    return V(succ, self.index + 1, self.depth)
    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing biased-expectimax

class BiasedExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your biased-expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the biased-expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing stop-biasedly from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def Vexpectimax(gameState, agent, depth):
      actions = gameState.getLegalActions(agent)
      if gameState.isWin() or gameState.isLose() or actions == []:
        return gameState.getScore(), None
      
      if depth == 0:
        return self.evaluationFunction(gameState), None
      
      if agent < gameState.getNumAgents() - 1: #next turn is ghost
        nextAgent = agent + 1
        nextDepth = depth
      else: # next turn is pacman
        nextAgent = 0
        nextDepth = depth - 1

      def p(a):
        if a == Directions.STOP: p = 0.5+0.5/len(actions)
        else: p = 0.5/len(actions)
        return p
      
      if agent == self.index: 
        return max((Vexpectimax(gameState.generateSuccessor(agent, a), nextAgent, nextDepth)[0], a) for a in actions)
      else:
        return sum(Vexpectimax(gameState.generateSuccessor(agent, a), nextAgent, nextDepth)[0]*p(a) for a in actions), None
      
    return Vexpectimax(gameState, self.index, self.depth)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    succ = gameState.generateSuccessor(0,action)
    def V(succ, agent, depth):
      actions = succ.getLegalActions(agent)
      if succ.isWin() or succ.isLose() or actions == []:
        return succ.getScore()
      
      if depth == 0:
        return self.evaluationFunction(succ)
      
      if agent < succ.getNumAgents() - 1:
        nextAgent = agent + 1
        nextDepth = depth
      else:
        nextAgent = 0
        nextDepth = depth - 1
      
      def p(a):
        if a == Directions.STOP: p = 0.5+0.5/len(actions)
        else: p = 0.5/len(actions)
        return p
      
      if agent == self.index: 
        return max(V(succ.generateSuccessor(agent, a), nextAgent, nextDepth) for a in actions)
      else:
        return sum(V(succ.generateSuccessor(agent, a), nextAgent, nextDepth)*p(a) for a in actions)
      
    return V(succ, self.index + 1, self.depth)
      
    # END_YOUR_ANSWER

######################################################################################
# Problem 4a: implementing expectiminimax

class ExpectiminimaxAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent (problem 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction

      The even-numbered ghost should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def Vexpectiminimax(gameState, agent, depth):
      actions = gameState.getLegalActions(agent)
      if gameState.isWin() or gameState.isLose() or actions == []:
        return gameState.getScore(), None
      
      if depth == 0:
        return self.evaluationFunction(gameState), None
      
      if agent < gameState.getNumAgents() - 1: #next turn is ghost
        nextAgent = agent + 1
        nextDepth = depth
      else: # next turn is pacman
        nextAgent = 0
        nextDepth = depth - 1

      
      if agent == self.index: 
        return max((Vexpectiminimax(gameState.generateSuccessor(agent, a), nextAgent, nextDepth)[0], a) for a in actions)
      elif agent%2 == 0: #agent is even
        return sum(Vexpectiminimax(gameState.generateSuccessor(agent, a), nextAgent, nextDepth)[0]/len(actions) for a in actions), None
      else: #agent is odd
        return min((Vexpectiminimax(gameState.generateSuccessor(agent, a), nextAgent, nextDepth)[0], a) for a in actions)

    return Vexpectiminimax(gameState, self.index, self.depth)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    succ = gameState.generateSuccessor(0, action)
    
    def V(succ, agent, depth):
      actions = succ.getLegalActions(agent)
      if succ.isWin() or succ.isLose() or actions == []:
        return succ.getScore()
      
      if depth == 0:
        return self.evaluationFunction(succ)
      
      if agent < succ.getNumAgents() - 1: #next turn is ghost
        nextAgent = agent + 1
        nextDepth = depth
      else: # next turn is pacman
        nextAgent = 0
        nextDepth = depth - 1

      
      if agent == self.index: 
        return max(V(succ.generateSuccessor(agent, a), nextAgent, nextDepth) for a in actions)
      elif agent%2 == 0: #agent is even
        return sum(V(succ.generateSuccessor(agent, a), nextAgent, nextDepth)/len(actions) for a in actions)
      else: #agent is odd
        return min(V(succ.generateSuccessor(agent, a), nextAgent, nextDepth) for a in actions)
    #print( V(gameState, self.index, self.depth)) 
    return V(succ, self.index + 1, self.depth)
    # END_YOUR_ANSWER

######################################################################################
# Problem 5a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent with alpha-beta pruning (problem 5)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER
    def alphabeta(gameState, agent, depth, alpha, beta):
      actions = gameState.getLegalActions(agent)
      
      if gameState.isWin() or gameState.isLose() or actions == []:
        return gameState.getScore(), None
      
      if depth == 0:
        return self.evaluationFunction(gameState), None
      
      if agent < gameState.getNumAgents() - 1: #next turn is ghost
        nextAgent = agent + 1
        nextDepth = depth
      else: # next turn is pacman
        nextAgent = 0
        nextDepth = depth - 1
        
      if agent == self.index:
        maxVal = float("-inf"), None
        for action in actions:
          if action == [] or action == None: continue
          maxVal = max(maxVal, (alphabeta(gameState.generatePacmanSuccessor(action), nextAgent, nextDepth, alpha, beta)[0], action))
          alpha = max(alpha, maxVal[0])
          if beta <= alpha: break
        return maxVal
      elif agent%2 == 0: #agent is even
        return sum(alphabeta(gameState.generateSuccessor(agent, a), nextAgent, nextDepth, alpha, beta)[0]/len(actions) for a in actions), None
      else:
        minVal = float("inf"), None
        for action in actions:
          if action == [] or action == None: continue
          minVal = min(minVal, (alphabeta(gameState.generateSuccessor(agent, action), nextAgent, nextDepth, alpha, beta)[0], action))
          beta = min(beta, minVal[0])
          if beta <= alpha: break
        return minVal
    #print(alphabeta(gameState, self.index, self.depth, float("-inf"), float("inf"))[0])
    return alphabeta(gameState, self.index, self.depth, float("-inf"), float("inf"))[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameStat, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    succ = gameStat.generateSuccessor(0, action)
    def Qalphabeta(succ, agent, depth, alpha, beta):
      actions = succ.getLegalActions(agent)
      
      if succ.isWin() or succ.isLose() or actions == []:
        return succ.getScore()
      
      if depth == 0:
        return self.evaluationFunction(succ)
      
      if agent < succ.getNumAgents() - 1: #next turn is ghost
        nextAgent = agent + 1
        nextDepth = depth
      else: # next turn is pacman
        nextAgent = 0
        nextDepth = depth - 1
        
      if agent == self.index:
        maxVal = float("-inf")
        for action in actions:
          if action == [] or action == None: continue
          maxVal = max(maxVal, (Qalphabeta(succ.generatePacmanSuccessor(action), nextAgent, nextDepth, alpha, beta)))
          alpha = max(alpha, maxVal)
          if beta <= alpha: break
        return maxVal
      elif agent%2 == 0: #agent is even
        return sum(Qalphabeta(succ.generateSuccessor(agent, a), nextAgent, nextDepth, alpha, beta)/len(actions) for a in actions)
      else:
        minVal = float("inf"), None
        for action in actions:
          if action == [] or action == None: continue
          minVal = min(minVal, (Qalphabeta(succ.generateSuccessor(agent, action), nextAgent, nextDepth, alpha, beta)))
          beta = min(beta, minVal)
          if beta <= alpha: break
        return minVal
    return Qalphabeta(succ, self.index+1, self.depth, float("-inf"), float("inf"))
    # END_YOUR_ANSWER

######################################################################################
# Problem 6a: creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 6).
  """

  # BEGIN_YOUR_ANSWER
  
  # default score
  features, weights = [], []
  #initialise with the original score
  features.append(currentGameState.getScore())
  weights.append(1)
  
  #1. number of foods
  nFood = currentGameState.getNumFood()
  features.append(nFood)
  weights.append(-0.1)
  
  #2. manhatten distance to the Scared ghost - good
  pPacman = currentGameState.getPacmanPosition()
  stateScaredGhosts = [ghost for ghost in currentGameState.getGhostStates() if ghost.scaredTimer]
  ScaredGhostDistance = [ghost.scaredTimer/manhattanDistance(pPacman, ghost.getPosition()) for ghost in stateScaredGhosts]
  features.append(min(ScaredGhostDistance) if len(ScaredGhostDistance) > 0 else 0)
  weights. append(20)
  

  #3. manhatten distance to the normal ghost - not good
  stateNormalGhosts = [ghost for ghost in currentGameState.getGhostStates() if not ghost.scaredTimer]
  normalGhostDistance = [manhattanDistance(pPacman, ghost.getPosition()) for ghost in stateNormalGhosts]
  features.append(min(normalGhostDistance) if len(normalGhostDistance) > 0 else 0)
  weights. append(-2)
  
  #4. manhattan distance to the food - small food --> have to move even if foodDistance big
  foodDistance = [manhattanDistance(pPacman, pFood) for pFood in currentGameState.getFood().asList()]
  features.append(((1 / min(foodDistance))) if len(foodDistance) > 0 else 0)
  weights.append(1/(nFood**2))

  # 5. manhattan distance to the capsule -- distancce big
  capsuleDistance = [1 / manhattanDistance(pPacman, pCapsule) for pCapsule in currentGameState.getCapsules()]
  features.append(min(capsuleDistance) if len(capsuleDistance) > 0 else 0)
  #print("distance:", len(ScaredGhostDistance))
  weights.append(5 if len(ScaredGhostDistance) == 0 else 0)

  # score
  return sum(feature * weight for feature, weight in zip(features, weights))
  # END_YOUR_ANSWER

def choiceAgent():
  """
    Choose the pacman agent model you want for problem 6.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
  """
  # BEGIN_YOUR_ANSWER
  return 'ExpectimaxAgent'
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction
