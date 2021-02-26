# Specifications

1. n-puzzle_solver: An n-puzzle is a sliding puzzle that consists of a frame of numbered square tiles in random order with one tile
missing. Popular sizes include a 3x3 board and a 5x5 board. The objective of the puzzle is to
place the tiles in order.
  The optimal sequence of moves can be obtained using search. For this machine problem, use A* search to get a
solution from the start to the goal board configuration.
Program Input
  Your program must accept a filename as commandline argument. The file shall contain the start state, followed
by the goal state. The blank tile shall be denoted by 0. <br/>
      7 2 4 <br/>
      5 0 6 <br/>
      8 3 1<br/>
      0 1 2<br/>
      3 4 5<br/>
      6 7 8<br/>
  The size of the puzzle may vary from 3x3 to 7x7.
  For simplicity, assume that the goal state is always reachable from the start state by some sequence of actions.
  Output through the terminal the optimal sequence of actions (up,down,left,right) to reach the goal state.
  
2. Clustering is the process of grouping objects in such a way that objects in a group are more similar to one
another than those from other groups. One application of clustering is in the automatic segmentation of regions
of interest in images. For this machine problem, you will apply kmeans clustering to isolate parasites in blood
smear images.

3. Facial recognition is one of many applications of machine learning. For this machine problem, you are to
implement a facial recognition system using artificial neural networks and support vector machines. <br/>
  DATASET: The dataset (http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) is composed of PGM
files of 40 subjects. Use 60% of the images from each subject for training and the remaining for testing.
You may use the pixel intensities as features.

Final Project: Dominant Colors Extraction <br/>
Used the k-means algorithm to extract the dominant colors from paintings <br/>
Programmers: Aguilana, Trina & Hernandez, Kat
