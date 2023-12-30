from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

## Example Agent
class ReflexAgent(Agent):

  def Action(self, gameState):

    move_candidate = gameState.getLegalActions()

    scores = [self.reflex_agent_evaluationFunc(gameState, action) for action in move_candidate]
    bestScore = max(scores)
    Index = [index for index in range(len(scores)) if scores[index] == bestScore]
    get_index = random.choice(Index)
    
    return move_candidate[get_index]

  def reflex_agent_evaluationFunc(self, currentGameState, action):

    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    return successorGameState.getScore()



def scoreEvalFunc(currentGameState):

  return currentGameState.getScore()

class AdversialSearchAgent(Agent):

  def __init__(self, getFunc ='scoreEvalFunc', depth ='2'):
    self.index = 0
    self.evaluationFunction = util.lookup(getFunc, globals())

    self.depth = int(depth)

class MinimaxAgent(AdversialSearchAgent):
  """
    [문제 01] MiniMaxAgent의 Action을 구현하시오.
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """
  def Action(self, gameState):
    ####################### Write Your Code Here ################################
    # getLegalActions(agentIndex) =>  ['North', 'South', 'West', 'East']
    # generateSuccessor(agentIndex, 'North') => gameState

    def Minimax(agentIndex, status, k):
      
      nextCandidate = status.getLegalActions(agentIndex) # if Win or Lose, return empty list 

      if len(nextCandidate) == 0 or k == self.depth: # 이기거나 졌거나, max depth에 도달했을 경우
        return (self.evaluationFunction(status), None)

      if agentIndex == 0: # meaning that agent is the pacman / max player
      
        Max = float("-inf")
        act = None

        for i in range(len(nextCandidate)):
          # 자식 노드들 중에서 최댓값을 가져온다.
          # 팩맨은 Max player 이기 때문에 자신의 이익을 극대화시키는 가능한 큰 점수를 선택한다.
          res = Minimax(agentIndex+1, status.generateSuccessor(agentIndex, nextCandidate[i]), k)
          if res[0] > Max:
            Max = res[0]
            act = nextCandidate[i]

        return (Max, act) # 최대일 때의 행동까지 튜플 형태로 반환

      else: # meaning that agent is the ghost / min player
        
        Min = float("inf")
        act = None
        # 고스트는 Min player 이기 때문에 팩맨의 이익을 극소화시키는 가능하면 작은 점수를 선택한다.

        if agentIndex == status.getNumAgents() - 1: # 현재 에이전트가 마지막 고스트이다.
          for i in range(len(nextCandidate)):
            # 마지막 고스트까지 Action을 취하면 depth 싸이클이 끝난 것이다.
            # 따라서, 다시 팩맨에게 차례를 넘겨주어야 한다. 첫 번째 인자로 팩맨의 인덱스인 0을 넘겨준다.
            # 한 depth 가 끝난 것이므로 depth 1 증가. k+1
            res = Minimax(0, status.generateSuccessor(agentIndex, nextCandidate[i]), k+1)
            if Min > res[0]:
              Min = res[0]
              act = nextCandidate[i]

        else:
          for i in range(len(nextCandidate)):
            # 다음 고스트에게 차례를 넘겨주어야 한다.
            res = Minimax(agentIndex+1, status.generateSuccessor(agentIndex, nextCandidate[i]), k)
            if Min > res[0]:
              Min = res[0]
              act = nextCandidate[i]

        return (Min, act) # 최소일 때의 행동까지 튜플 형태로 반환

    return Minimax(0, gameState, 0)[1] 

    raise Exception("Not implemented yet")

    ############################################################################



class AlphaBetaAgent(AdversialSearchAgent):
  """
    [문제 02] AlphaBetaAgent의 Action을 구현하시오.
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """
  def Action(self, gameState):
    ####################### Write Your Code Here ################################
    # getLegalActions(agentIndex) =>  ['North', 'South', 'West', 'East']
    # generateSuccessor(agentIndex, 'North') => gameState
    A = float("-inf") # 초기 알파값
    B = float("inf") # 초기 베타값

    def AlphaBetaPruning(agentIndex, status, k, alpha, beta):
      
      nextCandidate = status.getLegalActions(agentIndex) # if Win or Lose, return empty list 

      if len(nextCandidate) == 0 or k == self.depth: # 이기거나 졌거나, max depth에 도달했을 경우
        return (self.evaluationFunction(status), None)

      if agentIndex == 0: # meaning that agent is the pacman / max player
      
        Max = float("-inf")
        act = None

        for i in range(len(nextCandidate)):
          res = AlphaBetaPruning(agentIndex+1, status.generateSuccessor(agentIndex, nextCandidate[i]), k, alpha, beta)
          if res[0] > Max:
            Max = res[0]
            act = nextCandidate[i]

          if Max >= beta: # beta 보다 Max 값이 크거나 같으면, 더 이상 탐색할 필요가 없다. 위의 Min player는 어차피 beta 값을 유지할 것이기 때문이다.
            return (Max, act)
          alpha = max(alpha, Max) # Max player는 알파값을 업데이트한다. 

        return (Max, act)

      else: # meaning that agent is the ghost / min player
        
        Min = float("inf")
        act = None

        if agentIndex == status.getNumAgents() - 1: # 현재 에이전트가 마지막 고스트이다.
          for i in range(len(nextCandidate)):
            res = AlphaBetaPruning(0, status.generateSuccessor(agentIndex, nextCandidate[i]), k+1, alpha, beta) # 한 depth 가 끝난 것이므로 depth 1 증가
            if Min > res[0]:
              Min = res[0]
              act = nextCandidate[i]

            # 저장된 알파값보다 Min 값이 작거나 같으면, 더 이상 탐색할 필요가 없다. 위의 Max player는 어차피 alpha 값을 유지하게 되기 때문이다.
            if Min <= alpha: 
              return (Min, act)
            beta = min(beta, Min) # Min player는 베타값을 업데이트한다.

        else:
          for i in range(len(nextCandidate)):
            # 호출 방식은 MinimaxAgent과 동일
            res = AlphaBetaPruning(agentIndex+1, status.generateSuccessor(agentIndex, nextCandidate[i]), k, alpha, beta)
            if Min > res[0]:
              Min = res[0]
              act = nextCandidate[i]

            if Min <= alpha:
              return (Min, act)
            beta = min(beta, Min)

        return (Min, act)

    return AlphaBetaPruning(0, gameState, 0, A, B)[1]

    raise Exception("Not implemented yet")

    ############################################################################



class ExpectimaxAgent(AdversialSearchAgent):
  """
    [문제 03] ExpectimaxAgent의 Action을 구현하시오.
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """
  def Action(self, gameState):
    ####################### Write Your Code Here ################################
    # getLegalActions(agentIndex) =>  ['North', 'South', 'West', 'East']
    # generateSuccessor(agentIndex, 'North') => gameState

    def Expectimax(agentIndex, status, k):
      
      nextCandidate = status.getLegalActions(agentIndex) # if Win or Lose, return empty list 

      if len(nextCandidate) == 0 or k == self.depth: # 다음 선택지가 없거나, depth에 도달했을 때
        return (self.evaluationFunction(status), None)

      if agentIndex == 0: # meaning that agent is the pacman / max player
        Max = float("-inf")
        act = None

        for i in range(len(nextCandidate)):
          res = Expectimax(agentIndex+1, status.generateSuccessor(agentIndex, nextCandidate[i]), k)
          if res[0] > Max:
            Max = res[0]
            act = nextCandidate[i]
        
        return (Max, act)

      else: # meaning that agent is the ghost / min player
        Sum = 0
        if agentIndex == status.getNumAgents() - 1: # 현재 에이전트가 마지막 고스트이다.
          for i in range(len(nextCandidate)):
            # 확률을 곱한다는 것 외에 MinimaxAgent 구현과 다르지 않다.
            Sum += Expectimax(0, status.generateSuccessor(agentIndex, nextCandidate[i]), k+1)[0]

        else:
          for i in range(len(nextCandidate)):
            # 확률을 곱한다는 것 외에 MinimaxAgent 구현과 다르지 않다.
            Sum += Expectimax(agentIndex+1, status.generateSuccessor(agentIndex, nextCandidate[i]), k)[0]
          
        return (Sum / len(nextCandidate), None) # 기댓값을 반환 

    return Expectimax(0, gameState, 0)[1]

    raise Exception("Not implemented yet")
    ############################################################################
