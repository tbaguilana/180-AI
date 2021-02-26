# npuzzlesolver

An n-puzzle is a sliding puzzle that consists of a frame of numbered square tiles in random order with one tile
missing. Popular sizes include a 3x3 board and a 5x5 board. The objective of the puzzle is to
place the tiles in order.

The optimal sequence of moves can be obtained using search. For this machine problem, use A* search to get a
solution from the start to the goal board configuration.
Program Input
Your program must accept a filename as commandline argument. The file shall contain the start state, followed
by the goal state. The blank tile shall be denoted by 0.
7 2 4
5 0 6
8 3 1
0 1 2
3 4 5
6 7 8
The size of the puzzle may vary from 3x3 to 7x7.
For simplicity, assume that the goal state is always reachable from the start state by some sequence of actions.

Output through the terminal the optimal sequence of actions (up,down,left,right) to reach the goal state.
