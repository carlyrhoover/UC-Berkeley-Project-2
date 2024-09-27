# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        
        ##generates the successor state after the specific pacman move state = (0, action)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        
        ###returns a copy of the [0] index in the copied agentStates 
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ##ghostDistance = 0 
        pacmanPos = currentGameState.getPacmanPosition()
        newFood = currentGameState.getFood()
        newGhostStates = currentGameState.getGhostStates()
        currentScore = currentGameState.getScore()

        # Calculate the distance to the nearest food pellet
        minFood = 999999
        for x in range(newFood.width):
            for y in range(newFood.height):
                if (x, y) in newFood:
                    foodDistance = util.manhattanDistance(pacmanPos, (x, y))
                    if minFood > foodDistance:
                        minFood = foodDistance

        FOOD_WEIGHT = 10
        GHOST_WEIGHT = 100
        GHOST_MIN_DISTANCE = 2

    # Initialize evaluation
        evaluation = 0

    # Calculate the hypothetical successor game state after taking the action
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()

    # The following part is similar to your original code, but we incorporate the action evaluation
        foodScore = FOOD_WEIGHT / (minFood + 1)
        evaluation += foodScore  # Food collection is still encouraged

        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            ghostHeur = util.manhattanDistance(ghostPos, newPos)
            if ghostHeur <= GHOST_MIN_DISTANCE:
                evaluation -= GHOST_WEIGHT / (ghostHeur + 1)  # Avoiding ghosts is still discouraged

        evaluation += successorGameState.getScore()  # Incorporate the game score in the successor state

        return evaluation
            


def scoreEvaluationFunction(currentGameState: GameState):
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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        minAgents = gameState.getNumAgents() - 1  # Number of ghosts/min levels per turn

        def maxValue(gameState, depth, agentIndex):
            
            #worse casde for mac vale 
            maxScore = -99999 
            pacmanActions = gameState.getLegalActions(0)
            bestAction = None   

            ##if the game has been won, lost or depth has been reached 
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), bestAction
            
            if depth == self.depth:
                return self.evaluationFunction(gameState), bestAction

            for action in pacmanActions:
                successor = gameState.generateSuccessor(0, action)
                score, unused = minValue(successor, depth, agentIndex + 1)

                ##update the maximum score and the corresponding action and then return them 
                if score > maxScore:
                    maxScore = score
                    bestAction = action

            return maxScore, bestAction
        
        def minValue(gameState, depth, agentIndex):
            minScore = 99999  ##worst possibly case scenario
            bestAction = None
            ghostActions = gameState.getLegalActions(agentIndex)

            ##break if win lose or depth reached 
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), bestAction
            
            if depth == self.depth: 
                return self.evaluationFunction(gameState), bestAction
            
            ##iterate each legal ghost option 
            for action in ghostActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex < minAgents:
                    ##minimum score ghosts can achieve 
                    score, unused = minValue(successor, depth, agentIndex + 1)
                else:
                    ##pacman best value 
                    score, unused = maxValue(successor, depth + 1, 0)

                ###update to find the best mimum 
                if score < minScore:
                    minScore = score
                    bestAction = action

            return minScore, bestAction

        unused, bestAction = maxValue(gameState, 0, 0)

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        minAgents = gameState.getNumAgents() - 1  # Number of ghosts/min levels per turn

        ##pac man or maximizer value 
        def maxValue(gameState, depth, alpha, beta):
            maxScore = -999999 ##worst value 
            pacmanActions = gameState.getLegalActions(0)
            bestAction = None

            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), bestAction
            
            if depth == self.depth: 
                return self.evaluationFunction(gameState), bestAction

            ##for each action pacman can take chose the maximum
            for action in pacmanActions:
                successor = gameState.generateSuccessor(0, action)
            
                score, unused  = minValue(successor, depth, 1, alpha, beta)
                
                ##update if there is a better option 
                if score > maxScore:
                    maxScore = score
                    bestAction = action

                alpha = max(alpha, maxScore)  ##
                
                ##if the maxScore greater than the beta value, prune 
                if maxScore > beta:
                    break
 
            return maxScore, bestAction
        
        ##is very similar to maxvalue

        def minValue(gameState, depth, agentIndex, alpha, beta):
            minScore = 9999999
            bestAction = None
            ghostActions = gameState.getLegalActions(agentIndex)

            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), bestAction

            for action in ghostActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex < minAgents:
                    score, unused = minValue(successor, depth, agentIndex + 1, alpha, beta)
                else:
                    score, unused = maxValue(successor, depth + 1, alpha, beta)

                if score < minScore:
                    minScore = score
                    bestAction = action

                beta = min(beta, minScore)
                
                #prune 
                if minScore < alpha:
                    break

            return minScore, bestAction

        unused, bestAction = maxValue(gameState, 0, -99999, 99999999)
        return bestAction
        
       

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #just like max val of above
        
        
        minAgents = gameState.getNumAgents() - 1  # Number of ghosts/min levels per turn

        def maxValue(gameState, depth):
            maxScore = -999999
            pacmanActions = gameState.getLegalActions(0)
            bestAction = None

            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), bestAction

            if depth == self.depth:
                return self.evaluationFunction(gameState), bestAction

            for action in pacmanActions:
                successor = gameState.generateSuccessor(0, action)
                score, unused = expValue(successor, depth, 1)

                if score > maxScore:
                    maxScore = score
                    bestAction = action

            return maxScore, bestAction
        
        ##calculates the expected score; take the average of a chance node 
        def expValue(gameState, depth, agentIndex):
            bestAction = None
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), bestAction
            
            if depth == self.depth:
                return self.evaluationFunction(gameState), bestAction

            highestValue = 0
            legalActions = gameState.getLegalActions(agentIndex)

            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)

                
                if agentIndex < minAgents:
                    ##calls for the next agent 
                    score, unused = expValue(successor, depth, agentIndex + 1)
                else:
                    score, unused = maxValue(successor, depth + 1)

                numActions = len(legalActions)
                highestValue += score

            ##average score of chance found best value and corresponding action returned
            numActions = len(legalActions)
            highestValue /= numActions
            return highestValue, bestAction

        unused, bestAction = maxValue(gameState, 0)
        return bestAction
       

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    
    calculate the closest distance to a food pellet, incentivise that state until all food are gone
    Keep track of ghost states and their positions, states that the ghost distance is less than 1 decentivise 
    return the state that is of most value 
    
    """
    "*** YOUR CODE HERE ***"    
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    currentScore = currentGameState.getScore()

    # Initialize evaluation
    evaluation = currentScore 

    foodDistances = []

    for food in foodGrid:
        minFood = util.manhattanDistance(pacmanPos, food) 
        foodDistances.append(minFood)
    
    if len(foodDistances) != 0:
        minFoodDistance = min(foodDistances)
        evaluation += 10 / minFoodDistance  

    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        ghostDistance = util.manhattanDistance(pacmanPos, ghostPos)
        if ghostDistance <= 1:
            evaluation -= 1000  

    return evaluation
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
