# 导入相关包
import os
import random
import numpy as np
from Maze import Maze
from Runner import Runner
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot
import matplotlib.pyplot as plt

import numpy as np
from BFS import breadth_first_search
from AStar import a_star_search



# 迷宫路径搜索树



maze = Maze(maze_size=10)
height, width, _ = maze.maze_data.shape

# path_1 = breadth_first_search(maze)
path_1 = a_star_search(maze)
print("搜索出的路径：", path_1)

for action in path_1:
    maze.move_robot(action)

if maze.sense_robot() == maze.destination:
    print("恭喜你，到达了目标点")

print(maze)



