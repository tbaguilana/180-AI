# N-PUZZLE SOLVER

# An n-puzzle is a sliding puzzle that consists of a frame of numbered square tiles in random order with one tile
# missing. Popular sizes include a 3x3 board (shown below) and a 5x5 board. The objective of the puzzle is to
# place the tiles in order.

# The optimal sequence of moves can be obtained using search. For this machine problem, use A* search to get a
# solution from the start to the goal board configuration.

import sys	
import argparse
from heapq import heappush, heappop
import copy
import time

class Node():
	def __init__(self, boardConfig, hcost, gcost, parent):
		self.config = boardConfig
		self.fcost = hcost + gcost
		self.hcost = hcost
		self.parent = parent
		self.gcost = gcost
		self.move = " "

	def updateddata(self, hcost, gcost):
		self.fcost = hcost + gcost
		self.gcost = gcost
		self.hcost = hcost

	def __eq__(self, other):
		return self.retstring() == other.retstring()

	def __lt__(self, other):
		return self.fcost < other.fcost

	def retstring(self):
		string = ""
		for i in range(len(self.config)):
			string = string + str(self.config[i]) + "\n"
		return string
def menu():
	print "0. No Heuristic\n", "1. Misplaced Tiles\n", "2. Linear Conflict\n", "3. Tiles out of Row and Column\n", "4. N-Maxswap\n", "5. Manhattan Distance\n"
def readfile(filename):
	f = open(filename, 'r')
	string = f.read()
	newString = string.split('\n')
	
	f.close()
	return newString

def getInput(string):
	inputArray = []

	x = int(len(string)/2)

	inputArray = string[0:x]

	for i in range(len(inputArray)):
		inputArray[i] = inputArray[i].split(" ")
	
	for a in range(len(inputArray)):
		for b in range(len(inputArray)):
			inputArray[a][b] = int(inputArray[a][b])

	return inputArray

def getGoal(string):
	goalArray = []
	x = int(len(string)/2)
	y = int(len(string))
	
	goalArray = string[x:y]

	for i in range(len(goalArray)):
		goalArray[i] = goalArray[i].split(" ")
	
	for a in range(len(goalArray)):
		for b in range(len(goalArray)):
			goalArray[a][b] = int(goalArray[a][b])

	return goalArray

def heuristicpicker(curr, goal, heuristic):
	if heuristic == 0:
		curr.updateddata(0, curr.gcost+1)
	if heuristic == 1:
		curr.updateddata(misplaced(curr.config, goal.config), curr.gcost+1)
	if heuristic == 2:
		curr.updateddata(linearconflict(curr.config, goal.config), curr.gcost+1)
	if heuristic == 3:
		curr.updateddata(tilesout(curr.config, goal.config), curr.gcost+1)
	if heuristic == 4:
		curr.updateddata(nmax(curr.config, goal.config), curr.gcost+1)
	if heuristic == 5:
		curr.updateddata(manhattan(curr.config, goal.config), curr.gcost+1)

def expand(node, heuristic, goal):

	new_nodes = []
	c = None
	
	for x in range(len(node.config)):
		# print x
		for y in range(len(node.config)):
			# print y
			if node.config[x][y] == 0:
				c = (x,y)
				# print "yay"
				break

	if c[1] != len(node.config)-1:#can move left
		cpy = duplicate(node.config)
		h = node.hcost
		g = node.gcost
		copyNode = Node(cpy, h, g, node)
		copyNode.config[c[0]][c[1]] = copyNode.config[c[0]][c[1]+1]
		copyNode.config[c[0]][c[1]+1] = 0
		#copyNode.updateddata(h, copyNode.gcost+1)
		copyNode.move = "left"
		heuristicpicker(node, goal, heuristic)
		new_nodes.append(copyNode)

	if c[1] != 0: #can move right
		cpy = duplicate(node.config)
		h = node.hcost
		g = node.gcost
		copyNode = Node(cpy, h, g, node)
		copyNode.config[c[0]][c[1]] = copyNode.config[c[0]][c[1]-1]
		copyNode.config[c[0]][c[1]-1] = 0
		#copyNode.updateddata(h, copyNode.gcost+1)
		copyNode.move = "right"
		heuristicpicker(node, goal, heuristic)
		new_nodes.append(copyNode)
	
	
	if c[0] != len(node.config)-1: #can move up
		cpy = duplicate(node.config)
		h = node.hcost
		g = node.gcost
		copyNode = Node(cpy, h, g, node)
		copyNode.config[c[0]][c[1]] = copyNode.config[c[0]+1][c[1]]
		copyNode.config[c[0]+1][c[1]] = 0
		#copyNode.updateddata(h, copyNode.gcost+1)
		copyNode.move = "up"
		heuristicpicker(node, goal, heuristic)
		new_nodes.append(copyNode)

	if c[0] != 0: #can move down
		cpy = duplicate(node.config)
		h = node.hcost
		g = node.gcost
		copyNode = Node(cpy, h, g, node)
		copyNode.config[c[0]][c[1]] = copyNode.config[c[0]-1][c[1]]
		copyNode.config[c[0]-1][c[1]] = 0
		#copyNode.updateddata(h, copyNode.gcost+1)
		copyNode.move = "down"
		heuristicpicker(node, goal, heuristic)
		new_nodes.append(copyNode)

	return new_nodes

def printMoves(node):
	moves = []
	new = []
	while node.parent is not None:
		# print "smth"
		new.append(node.retstring())
		moves.append(node.move)
		node = node.parent

	moves.reverse()
	new.reverse()
	# for each in new:
	# 	print each
	for each in moves:
		print each
	# print len(moves)

def duplicate(config):
	cpy = []
	for i in range(len(config)):
		cpy.append([])
		for j in range(len(config)):
			cpy[i].append(config[i][j])

	return cpy

def ihash(config):
	curr = {}
	for i in range(len(config)):
		for j in range(len(config)):
			curr[config[i][j]] = (i,j)

	return curr

def newHash(config):
	curr = {}
	for i in range(len(config)):
		curr[config[i]] = i
	return curr

def tilesout(config, goal):
	count = 0
	test = ihash(goal)
	for i in range(len(config)):
		for j in range(len(config)):
			if config[i][j] != 0:
				if test[config[i][j]][0]  != i:
					count = count + 1
				if test[config[i][j]][1] != j:
					count = count + 1

	return count

def misplaced(config, goal):
	count = 0
	for i in range(len(config)):
		for j in range(len(config)):
			if config[i][j] != 0 and config[i][j] != goal[i][j]:
				count = count + 1

	return count

def manhattan(config, goal):
	count = 0
	test = ihash(goal)
	for i in range(len(config)):
		for j in range(len(config)):
			if config[i][j] != 0:
				dx = abs(test[config[i][j]][0] - i)
				dy = abs(test[config[i][j]][1] - j) 
				count = count + dx + dy
				

	return count

def linearconflict(config, goal):
	count = manhattan(config, goal)
	temp = 0
	test = ihash(goal)
	#row conflict
	for i in range(len(config)):
		for j in range(len(config)):
			for k in range(j+1, len(config)):
				tj = config[i][j]
				tk = config[i][k]
				if test[tk][1] < test[tj][1]:
					if test[tk][0] == i and test[tk][0] == i:
						if tk != 0 and tj != 0:
							temp = temp + 1
	#column conflict
	for i in range(len(config)):
		for j in range(len
			(config)):
			for k in range(i+1, len(config)):
				tj = config[i][j]
				tk = config[k][j]
				if test[tk][0] < test[tj][0]:
					if test[tk][1] == j and test[tk][1] == j:
						if tk != 0 and tj != 0:
							temp = temp + 1
	count += 2*temp
	return count

def stringcomp(config, goal):
	# print goal
	string1 = ""
	for i in range(len(config)):
		string1 = string1 + str(config[i]) + " "

	string2 = ""
	for j in range(len(goal)):
		string2 = string2 + str(goal[j]) + " "

	# print string1
	# print string2
	# string3 = ""
	# for k in range(len(config)):
	# 	string3 = string3 + str(config[i]) + "\n"

	return string1 != string2


def nmax(config, goal):
	count = 0
	newconfig = [] #P
	newgoal = []
	B = []
	test = []
	for i in range(len(config)):
		newconfig = newconfig + config[i]
		newgoal = newgoal + goal[i]

	for m in range(len(newconfig)):
		B.append(0)
		test.append(0)
	
	# print newconfig
	B = newHash(newconfig)

	while stringcomp(newconfig, B.values()):
		count = count + 1
		newconfig[B[0]] , newconfig[B[B[0]]] = newconfig[B[B[0]]] ,newconfig[B[0]]
		# print newconfig
		B = newHash(newconfig)
		# print B.values()
		
	return count
def main():
	menu()
	
	openSet = []
	openHash = {}
	closedSet = {}
	filename = sys.argv[1]
	inputString = readfile(filename)
	
	startState = []
	startState = getInput(inputString)

	goalState = []
	goalState = getGoal(inputString)

	choice_heuristic = int(raw_input())
	starttime = time.time()
	start = Node(startState, 0, 0, None)
	end = Node(goalState, 0, 0, None)
	heuristicpicker(start, end, choice_heuristic)
	# print end.retstring()
	heappush(openSet, start)
	openHash[start.retstring()] = start
	states = 0
	
	# nmax(startState, goalState)
	while(len(openSet) != 0):
		states = states + 1
	
		curr = heappop(openSet)
		del openHash[curr.retstring()]
		
		if curr.retstring() == end.retstring():
			# print curr.retstring()
			printMoves(curr)
			break

		closedSet[curr.retstring()] = curr

		children = expand(curr, choice_heuristic, end)
		# print curr.config
		for each in children:

			try: 
				closedSet[each.retstring()]
				continue
			except KeyError:
				pass

			try:
				openHash[each.retstring()]
				if each.gcost < openHash[each.retstring()].gcost:
					openHash[each.retstring()].gcost = each.gcost
					openHash[each.retstring()].fcost = each.fcost
					openHash[each.retstring()].parent = each.parent
			except KeyError:
				heappush(openSet, each)
				openHash[each.retstring()] = each

	# print states
	# print "%s seconds" %(time.time() - starttime)

main()