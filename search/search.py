# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import math

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    mystack = util.Stack()
    startNode = (problem.getStartState(), '', 0, [])
    mystack.push(startNode)
    visited = set()
    while mystack :
        node = mystack.pop()
        state, action, cost, path = node
        if state not in visited :
            visited.add(state)
            if problem.isGoalState(state) :
                path = path + [(state, action)]
                break;
            succNodes = problem.expand(state)
            for succNode in succNodes :
                succState, succAction, succCost = succNode
                newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                mystack.push(newNode)
    actions = [action[1] for action in path]
    del actions[0]
    return actions

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

def aStarSearch(problem, heuristic=nullHeuristic):
    #COMP90054 Task 1, Implement your A Star search algorithm here
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    mystack = util.PriorityQueue()
    startNode = (problem.getStartState(), '', 0, [])
    mystack.push(startNode, heuristic(problem.getStartState(), problem))
    visited = set()
    while mystack:
        node = mystack.pop()
        state, action, cost, path = node
        if state not in visited:
            visited.add(state)
            if problem.isGoalState(state):
                break;
            succNodes = problem.expand(state)
            for succNode in succNodes:
                succState, succAction, succCost = succNode
                succ_path = path + [(succState, succAction)]
                actions = [action[1] for action in succ_path]
                newNode = (succState, succAction, problem.getCostOfActionSequence(actions), succ_path)
                mystack.update(newNode,heuristic(succState,problem)+newNode[2])
    actions = [action[1] for action in path]
    return actions

        
# def recursivebfs(problem, heuristic=nullHeuristic) :
#     #COMP90054 Task 2, Implement your Recursive Best First Search algorithm here
#     "*** YOUR CODE HERE ***"
#
#     def RBFS(problem, node, flimit):
#        # state, action, cost, path, f = node
#         if problem.isGoalState(node[0]):
#             node[3]= node[3] + [(node[0],node[1])]
#             actions = [action[1] for action in node[3]]
#             del actions[0]
#             return actions, 0
#         successors = problem.expand(node[0])
#         if len(successors) == 0:
#             # actions = [action[1] for action in node[3]]
#             # del actions[0]
#             return [], math.inf
#         for index in range(len(successors)):
#             succState, succAction, succCost = successors[index]
#             # newS = {'state':succState, 'action':succAction, 'cost':node['cost'] + succCost,
#             #         'path':node['path'] + [(node['state'], node['action'])],'f':max(heuristic(succState,problem)+node['cost'] + succCost,node["f"])}
#             newS = [succState, succAction, node[2] + succCost,
#                     node[3] + [(node[0], node[1])],max(heuristic(succState,problem)+node[2] + succCost,node[4])]
#             successors[index] = newS
#         while True:
#             # Order by lowest f value
#             successors.sort(key=lambda s: s[4])
#             best = successors[0]
#             if best[4] > flimit:
#                 # best[3] = best[3] + [(best[0], best[1])]
#                 # actions = [action[1] for action in best[3]]
#                 # del actions[0]
#                 return [], best[4]
#             if len(successors) > 1:
#                 alternative = successors[1]
#                 alternativestate, alternativeaction, alternativecost, alternativepath, alternativef = alternative
#             else:
#                 alternativef = math.inf
#             result, best[4] = RBFS(problem, best, min(flimit, alternativef))
#             if result is not []:
#                 return result, best[4]
#             else:
#                 return [],math.inf
#
#     startNode = [problem.getStartState(), '', 0, [],heuristic(problem.getStartState(),problem)]
#     result, bestf = RBFS(problem, startNode, math.inf)
#     return result


def recursivebfs(problem, heuristic=nullHeuristic) :
    startNode = [problem.getStartState(), '', 0, [], heuristic(problem.getStartState(), problem)]
    result , num = RBFS(problem,startNode,math.inf,heuristic)
    return result

def RBFS(problem,node,f_limit,heuristic):
    mystack = util.PriorityQueue()
    if problem.isGoalState(node[0]):
        path = node[3] + [(node[0],node[1])]
        actions = [action[1] for action in path]
        del actions[0]
        return actions , 0
    successors = problem.expand(node[0])
    if len(successors) == 0:
        return "failure",math.inf
    for successor in successors:
        succState, succAction, succCost = successor
        newNode = [succState,succAction,node[2]+succCost,node[3] + [(node[0], node[1])],
                   max(heuristic(succState, problem) + node[2] + succCost, node[4])]
        mystack.update(newNode,max(heuristic(succState, problem) + node[2] + succCost, node[4]))
    while True:
        best = mystack.pop()
        if best[4] > f_limit:
            return "failure",best[4]
        if mystack.isEmpty():
            alternative_f = math.inf
        else:
            alternative = mystack.pop()
            alternative_f = alternative[4]
            mystack.update(alternative,alternative[4])
        result, best[4] = RBFS(problem,best,min(f_limit,alternative_f),heuristic)
        mystack.update(best,best[4])
        if result != "failure":
            return result,0

    
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
rebfs = recursivebfs
