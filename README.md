# Artifical-intellegence-Assignment
#1)Apply BFS and DFS on trees and graphs. You can use simple examples for your practice. For 
graph editor you can use this:

TASK 1: Implementations of BFS adnd DFS

from collections import deque


class TreeNode:
    def _init_(self, value):
        self.value = value
        self.children = []


def bfs_tree(root):
    if not root:
        return

    queue = deque()
    visited = set()

    queue.append(root)
    visited.add(root)

    while queue:
        node = queue.popleft()
        print(node.value)

        for child in node.children:
            if child not in visited:
                queue.append(child)
                visited.add(child)


def dfs_tree(node):
    if not node:
        return

    print(node.value)

    for child in node.children:
        dfs_tree(child)


def bfs_graph(graph, start):
    if not graph or start not in graph:
        return

    queue = deque()
    visited = set()

    queue.append(start)
    visited.add(start)

    while queue:
        node = queue.popleft()
        print(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)


def dfs_graph(graph, start, visited=None):
    if not graph or start not in graph:
        return

    if visited is None:
        visited = set()

    print(start)
    visited.add(start)

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs_graph(graph, neighbor, visited)


# Example usage:

# Create a tree
root = TreeNode('A')
node1 = TreeNode('B')
node2 = TreeNode('C')
node3 = TreeNode('D')
node4 = TreeNode('E')

root.children = [node1, node2]
node1.children = [node3]
node2.children = [node4]

print("BFS on Tree:")
bfs_tree(root)

print("DFS on Tree:")
dfs_tree(root)

print("BFS on Graph:")
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
bfs_graph(graph, 'A')

print("DFS on Graph:")
dfs_graph(graph, 'A')
     
BFS on Tree:
A
B
C
D
E
DFS on Tree:
A
B
D
C
E
BFS on Graph:
A
B
C
D
E
F
DFS on Graph:
A
B
D
E
F
C


TASK 2: Apply BFS and DFS
Certainly! Below is the Python code that generates random and unique numbers, builds trees, applies BFS and DFS, and creates a data frame with the execution times:


import random
import time
import pandas as pd

# Generate random and unique numbers for each set
def generate_unique_numbers(num_elements):
    return random.sample(range(1, 1000001), num_elements)

# Define a binary search tree node
class TreeNode:
    def _init_(self, value):
        self.value = value
        self.left = None
        self.right = None

# Build a binary search tree from a list of numbers
def build_tree(numbers):
    root = None
    for num in numbers:
        root = insert_node(root, num)
    return root

# Insert a node into the binary search tree
def insert_node(root, value):
    if root is None:
        return TreeNode(value)
    if value < root.value:
        root.left = insert_node(root.left, value)
    else:
        root.right = insert_node(root.right, value)
    return root

# Perform Breadth-First Search (BFS) to find the goal
def bfs(root, goal):
    queue = [root]
    start_time = time.time()
    while queue:
        node = queue.pop(0)
        if node.value == goal:
            return time.time() - start_time
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

# Perform Depth-First Search (DFS) to find the goal
def dfs(root, goal):
    stack = [root]
    start_time = time.time()
    while stack:
        node = stack.pop()
        if node.value == goal:
            return time.time() - start_time
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

# Generate random numbers for each set
set1 = generate_unique_numbers(220)
set2 = generate_unique_numbers(220)
set3 = generate_unique_numbers(220)
set4 = generate_unique_numbers(220)

# Build trees for each set
tree1 = build_tree(set1)
tree2 = build_tree(set2)
tree3 = build_tree(set3)
tree4 = build_tree(set4)

# Find the goal (lis[total_len – 220])
total_len = 220
goal = set1[total_len - 220]
# Calculate BFS and DFS times for each set
bfs_time_set1 = bfs(tree1, goal)
dfs_time_set1 = dfs(tree1, goal)

bfs_time_set2 = bfs(tree2, goal)
dfs_time_set2 = dfs(tree2, goal)

bfs_time_set3 = bfs(tree3, goal)
dfs_time_set3 = dfs(tree3, goal)

bfs_time_set4 = bfs(tree4, goal)
dfs_time_set4 = dfs(tree4, goal)

# Create a data frame
df = pd.DataFrame({
    "Tree Size": [1000, 40000, 80000, 200000, 1000000],
    "BFS Time": [bfs_time_set1, bfs_time_set2, bfs_time_set3, bfs_time_set4, None],
    "DFS Time": [dfs_time_set1, dfs_time_set2, dfs_time_set3, dfs_time_set4, None]
})

# Print the data frame
print(df)

     
   Tree Size  BFS Time      DFS Time
0       1000  0.000005  7.152557e-07
1      40000       NaN           NaN
2      80000       NaN           NaN
3     200000       NaN           Na



A* Algorithm maze using python
To solve the maze using the A* search algorithm in Python, we’ll need to define the maze as a grid, where each cell is either a wall or a passable node. We’ll represent walls with a value of 1 and passable nodes with a value of 0. The start and goal nodes will be represented by their coordinates.

Here’s a Python code snippet that implements the A* search algorithm:


import heapq

class Node:
    def _init_(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def _eq_(self, other):
        return self.position == other.position

    def _lt_(self, other):
        return self.f < other.f

def astar(maze, start, end):
    # Create start and end node
    start_node = Node(start, None)
    end_node = Node(end, None)

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    heapq.heappush(open_list, start_node)

    # Loop until you find the end
    while len(open_list) > 0:
        # Get the current node
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # Adjacent squares
            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) - 1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(node_position, current_node)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            heapq.heappush(open_list, child)

    return None

# Define the maze as a list of lists
maze = [
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0]
]

# Define start and end points
start = (0, 0)  # Assuming 'A' is at the top-left corner
end = (5, 5)   # Assuming 'Y' is at the bottom-right corner

path = astar(maze, start, end)
print(path)

     
[(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (4, 3), (5, 3), (5, 4), (5, 5)]


Alpha-Beta Pruning implementations using python

class Node:
    def _init_(self, value):
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)


def alpha_beta(node, depth, alpha, beta, maximizing_player):
    if depth == 0 or len(node.children) == 0:
        return node.value

    if maximizing_player:
        value = float('-inf')
        for child in node.children:
            value = max(value, alpha_beta(child, depth - 1, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float('inf')
        for child in node.children:
            value = min(value, alpha_beta(child, depth - 1, alpha, beta, True))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value
A = Node(3)
B = Node(5)
C = Node(2)
D = Node(9)
E = Node(1)
F = Node(8)
G = Node(4)

A.add_child(B)
A.add_child(C)
A.add_child(D)
B.add_child(E)
B.add_child(F)
D.add_child(G)

result = alpha_beta(A, 3, float('-inf'), float('inf'), True)
print("Best move score:", result)
     
Best move score: 4




