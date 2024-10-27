import heapq
import numpy as np

# Robot movement directions
move_map = {
    'u': (-1, 0),  # up
    'r': (0, +1),  # right
    'd': (+1, 0),  # down
    'l': (0, -1),  # left
}

class SearchTree(object):
    def __init__(self, loc=(), action='', parent=None):
        """
        Initialize a search tree node object
        :param loc: The location of the robot at the new node
        :param action: The move direction corresponding to the new node
        :param parent: The parent node of the new node
        """
        self.loc = loc  # Current node location
        self.to_this_action = action  # Action to reach the current node
        self.parent = parent  # Parent node of the current node
        self.children = []  # Children nodes of the current node
        self.priority = 0  # Priority of the node

    def add_child(self, child):
        """
        Add a child node
        :param child: The child node to be added
        """
        self.children.append(child)

    def is_leaf(self):
        """
        Check if the current node is a leaf node
        """
        return len(self.children) == 0

    def __lt__(self, other):
        """
        Compare nodes based on priority
        """
        return self.priority < other.priority

def back_propagation(node):
    """
    Backtrack and record the node path
    :param node: The node to backtrack from
    :return: The backtracked path
    """
    path = []
    while node.parent is not None:
        path.insert(0, node.to_this_action)
        node = node.parent
    return path

def expand(maze, is_visit_m, node):
    """
    Expand leaf nodes, i.e., add child nodes reached by performing legal actions from the current leaf node
    :param maze: The maze object
    :param is_visit_m: Matrix recording whether each position in the maze has been visited
    :param node: The leaf node to be expanded
    """
    can_move = maze.can_move_actions(node.loc)
    for a in can_move:
        new_loc = tuple(node.loc[i] + move_map[a][i] for i in range(2))
        if not is_visit_m[new_loc]:
            child = SearchTree(loc=new_loc, action=a, parent=node)
            node.add_child(child)

def heuristic(loc, goal):
    """
    Calculate the heuristic function value, here using Manhattan distance
    :param loc: Current node location
    :param goal: Goal location
    :return: Heuristic function value
    """
    return abs(loc[0] - goal[0]) + abs(loc[1] - goal[1])

def a_star_search(maze):
    """
    Perform A* search on the maze
    :param maze: The maze object to be searched
    :return: The path found by A* search
    """
    start = maze.sense_robot()  # Get the start location
    goal = maze.destination  # Get the goal location
    root = SearchTree(loc=start)  # Create the root node
    open_list = []  # Priority queue for open nodes
    root.priority = 0 + heuristic(start, goal)  # Set the priority for the root
    heapq.heappush(open_list, (root.priority, 0, root))  # Push the root node into the priority queue
    h, w, _ = maze.maze_data.shape
    is_visit_m = np.zeros((h, w), dtype=np.int32)  # Matrix to record visited positions
    g_costs = {start: 0}  # Dictionary to record the actual cost from the start to each point
    path = []  # List to record the path

    while open_list:
        _, current_cost, current_node = heapq.heappop(open_list)  # Pop the node with the lowest priority
        is_visit_m[current_node.loc] = 1  # Mark the current node location as visited

        if current_node.loc == goal:  # If the goal is reached
            path = back_propagation(current_node)  # Backtrack to get the path
            break

        if current_node.is_leaf():
            expand(maze, is_visit_m, current_node)  # Expand the current node if it is a leaf

        for child in current_node.children:
            new_cost = current_cost + 1  # Assume the cost of each step is 1
            if new_cost < g_costs.get(child.loc, float('inf')):
                g_costs[child.loc] = new_cost  # Update the cost to reach the child node
                child.priority = new_cost + heuristic(child.loc, goal)  # Set the priority for the child
                heapq.heappush(open_list, (child.priority, new_cost, child))  # Push the child node into the priority queue

    return path